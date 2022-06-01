class DP(nn.Module):
    def __init__(self, word_to_idx, pos_to_idx, word_embed, pos_embed, word_embed_len, pos_embed_len,
                 feature_vec_len=20, hidden_size=400, nn_arc_hidden_size=500,
                 nn_label_hidden_size=200, n_labels=47):
        super(DP, self).__init__()
        self.word_embed_len = word_embed_len
        self.pos_embed_len = pos_embed_len
        self.data_vec_len = word_embed_len + pos_embed_len
        self.feature_vec_len = feature_vec_len
        self.hidden_size = hidden_size
        self.nn_arc_hidden_size = nn_arc_hidden_size
        self.nn_label_hidden_size = nn_label_hidden_size
        self.n_labels = n_labels
        self.word_to_idx = word_to_idx
        self.pos_to_idx = pos_to_idx

        # declaring parameters
        self.word_embed = torch.nn.Embedding(len(word_embed), word_embed_len)
        self.pos_embed = torch.nn.Embedding(len(pos_embed), pos_embed_len)
        self.word_embed.weight = torch.nn.Parameter(word_embed)
        self.pos_embed.weight = torch.nn.Parameter(pos_embed)

        self.BiLSTM = torch.nn.LSTM(input_size=self.data_vec_len, hidden_size=self.hidden_size, num_layers=3,
                                    dropout=0.3, bidirectional=True)

        # arch head nn layers
        self.arc_head_linear1 = nn.Linear(self.hidden_size * 2, self.nn_arc_hidden_size)
        self.arc_head_linear2 = nn.Linear(self.nn_arc_hidden_size, self.feature_vec_len)

        # arch dep nn layer
        self.arc_dep_linear1 = nn.Linear(self.hidden_size * 2, self.nn_arc_hidden_size)
        self.arc_dep_linear2 = nn.Linear(self.nn_arc_hidden_size, self.feature_vec_len)

        # head label layer
        self.label_head_linear1 = nn.Linear(self.hidden_size * 2, self.nn_label_hidden_size)
        self.label_head_linear2 = nn.Linear(self.nn_label_hidden_size, self.feature_vec_len)

        # dep label layer
        self.label_dep_linear1 = nn.Linear(self.hidden_size * 2, self.nn_label_hidden_size)
        self.label_dep_linear2 = nn.Linear(self.nn_label_hidden_size, self.feature_vec_len)

        # label classifier
        self.label_class_hidden = nn.Linear(self.feature_vec_len * 2, self.feature_vec_len)
        self.output = nn.Linear(self.feature_vec_len, self.n_labels)

        self.u1 = nn.Parameter(torch.rand(feature_vec_len, feature_vec_len))
        self.u2 = nn.Parameter(torch.rand(1, feature_vec_len))

        self.arc_loss = []
        self.label_loss = []
        self.total_loss = []

    def arc_head(self, x):
        temp = F.relu(self.arc_head_linear1(x))
        return self.arc_head_linear2(temp)

    def arc_dep(self, x):
        temp = F.relu(self.arc_dep_linear1(x))
        return self.arc_dep_linear2(temp)

    def label_head(self, x):
        temp = F.relu(self.label_head_linear1(x))
        return self.label_head_linear2(temp)

    def label_dep(self, x):
        temp = F.relu(self.label_dep_linear1(x))
        return self.label_dep_linear2(temp)

    def label_classifier(self, x):
        temp = F.relu(self.label_class_hidden(x))
        return self.output(temp)

    def forward(self, xdata):
        x_len = len(xdata[0])
        word_seq = xdata[:, 0]
        pos_seq = xdata[:, 1]
        gold_tree = xdata[:, 2] if x_len == 3 else None

        # fetch word embeds
        word_embdding = self.word_embed(word_seq)
        pos_embedding = self.pos_embed(pos_seq)

        temp = torch.cat((word_embdding, pos_embedding), 1)
        temp = temp[:, None, :]  # adding an empty dimension for lstm input

        # initialize hidden layers
        h1 = torch.zeros(6, 1, self.hidden_size)
        h2 = torch.zeros(6, 1, self.hidden_size)
        hidden = (autograd.Variable(h1), autograd.Variable(h2))

        # embed words in their context
        c, _ = self.BiLSTM(temp, hidden)

        # calculate arcs
        arc_head_ = torch.squeeze(self.arc_head(c))
        arc_dep_ = torch.squeeze(self.arc_dep(c))
        adj_mat = arc_head_ @ self.u1 @ torch.t(arc_dep_) + arc_head_ @ torch.t(self.u2)

        # calculate labels, if gold tree is not None = train else test
        pred_labels = None
        if gold_tree is not None:
            label_head_ = torch.squeeze(self.label_head(c))
            label_dep_ = torch.squeeze(self.label_dep(c))
            label_dep_ = label_dep_[gold_tree.data]
            arc_to_label = torch.cat((label_head_, label_dep_), 1)
            pred_labels = self.label_classifier(arc_to_label)

        return adj_mat, pred_labels