import os
import json
import glob
import copy
import torch
import pickle
import time

def convert_conllu_to_json_file(filepath=None):

    # initialize
    text = []

    # Read conllu file
    with open(filepath, 'r', encoding='utf-8') as f_handle:
        source = f_handle.read()

    sentences = source.strip().split('\n\n')

    for s in sentences:
        temp_lines = s.strip().split('\n')

        sent = {}
        lines = []
        for l in temp_lines:
            words = l.split(' ')
            # Drop all lines beginning with #
            if words[0] == '#':
                if words[1] == 'sent_id':
                    sent['sent_id'] = words[3]
            else:
                lines.append(l)

        reject_sentence = False
        words = []
        for l in lines:
            words_list = l.split('\t')

            try:
                int(words_list[0])
            except ValueError:
                reject_sentence = True
                break

            word = {
                "id": words_list[0],
                "form": words_list[1],
                "lemma": words_list[2],
                "upostag": words_list[3],
                "xpostag": words_list[4],
                "feats": words_list[5],
                "head": words_list[6],
                "deprel": words_list[7],
                "deps": words_list[8],
                "misc": words_list[9]
            }
            words.append(word)

            if word['deprel'] == 'root':
                words.append({
                    "id": "0",
                    "form": "<ROOT> ",
                    "lemma": "<ROOT>",
                    "upostag": "ROOT",
                    "xpostag": "ROOT",
                    "feats": "_",
                    "head": "-1",
                    "deprel": "_",
                    "deps": "_",
                    "misc": "_"
                })

        if reject_sentence:
            continue

        sent['words'] = words
        text.append(sent)

    with open(filepath.replace('conllu', 'json'), 'w+') as f_handle:
        f_handle.write(json.dumps(text, indent=4))


def preprocess_data(filep):
    dump_file_path = "/content/drive/MyDrive/Dependency_Parsing/UD_English-Atis/en_atis-ud-train_unk.json"
    text = json.load(open(filep, 'r'))
    words = {}
    data = []

    max_length = 15

    for t in text:
        if len(t['words']) <= max_length:
            data.append(t)

    for t in data:
        for w in t['words']:
            freq = words.get(w['form'])
            if freq is None:
                words[w['form']] = 1
            else:
                words[w['form']] = freq + 1

    single_occ_words = []
    for w in words.keys():
        if words[w] == 1:
            single_occ_words.append(w)

    # dump this data
    for t in data:
        for w in t['words']:
            if w['form'] in single_occ_words:
                w['form'] = '<unk>'

    with open(dump_file_path, 'w') as file_handle:
        file_handle.write(json.dumps(data, indent=4))