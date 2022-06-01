def create_test_data(trainfile, testfile, label_file):
    train_data = json.load(open(trainfile, 'r'))
    words = {}

    # count occurences
    for s in train_data:
        for w in s['words']:
            oc = w.get(w['form'])
            if oc is None:
                words[w['form']] = 1
            else:
                words[w['form']] = oc + 1

    multi_words = []
    for w in words.keys():
        if words[w] > 1:
            multi_words.append(w)

    test_data = json.load(open(testfile, 'r'))

    # create new data
    for s in test_data:
        for w in s['words']:
            if w['form'] not in multi_words:
                w['form'] = '<unk>'

    labels_in_trainset = json.load(open(labelsfile, 'r'))

    test_data_with_existing_labels = []

    for obj in test_data:
        ad = True
        for w in obj['words']:
            if word['form'] == "Administrator":
                add = False
                break
            if word['deprel'] not in labels_in_trainset.keys():
                add = False
                break
        if add:
            test_data_with_existing_labels.append(obj)

    with open(test_preprocessed_file, 'w+') as f_handle:
        f_handle.write(json.dumps(test_data_with_existing_labels, indent=4))


def evaluate(model, test_data):
    model.eval()
    arc_losses = []
    label_losses = []
    total_losses = []

    for i in range(len(test_data)):

        # fetch the length of each sequence
        seq_len = len(data[i]['words'])

        # create the adj matrix and tree
        gold_mat = convert_seq_to_mat(data[i])
        gold_tree = adj_mat_to_tensor(gold_mat)

        arc_target = Variable(gold_tree, requires_grad=False)
        labels_target = torch.LongTensor(seq_len)

        for j, w in enumerate(data[i]['words']):
            labels_target[j] = lab_to_idx[w['deprel']]
        labels_target = Variable(labels_target, requires_grad=False)

        # prepare input
        seq = torch.LongTensor(seq_len, 3)
        for j, w in enumerate(data[i]['words']):
            seq[j, 0] = word_to_idx[w['form']]
            seq[j, 1] = pos_to_idx[w['xpostag']]
            seq[j, 2] = gold_tree[j]
        seq_var = Variable(seq)

        # run the model
        adj_mat, label_pred = model(seq_var)

        # determine losses
        arc_pred = torch.t(adj_mat)  # cross entropy loss wants the classes in the second dimension
        arc_loss = criterion(arc_pred, arc_target)
        label_loss = criterion(label_pred, labels_target)
        total_loss = arc_loss + label_loss
        arc_losses.append(arc_loss.item())
        label_losses.append(label_loss.item())
        total_losses.append(total_loss.item())

    arc_loss = np.mean(arc_losses)
    label_loss = np.mean(label_losses)
    total_loss = np.mean(total_losses)

    print("ARC Loss on test data : " + str(arc_loss))
    print("Label Loss on test data : " + str(label_loss))
    print("Total Loss on test data : " + str(total_loss))