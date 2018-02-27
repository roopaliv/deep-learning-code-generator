from __future__ import division, print_function
import numpy as np
import os
from collections import OrderedDict
import string

# This file will be responsible of preprocessing the input data
# Creation of word embeddings
# Probabilistic Distribution of the sequence of texts

FIRST_INDEX = ord('a')
#vocab = OrderedDict((k,v+1) for v,k in enumerate(string.printable))
vocab = {}
inverse_vocab = {}

#inverse_vocab = {v: k for k, v in vocab.items()}

def get_vocab(unq_chars):
    vocab = dict((v, k) for k, v in enumerate(unq_chars))
    return vocab

def get_inverse_vocab(unq_chars):
    inverse_vocab = dict((k, v) for k, v in enumerate(unq_chars))
    return inverse_vocab


def get_text(data_dir):
    file_list = []
    for files in os.listdir(data_dir):
        if files.endswith('.cpp'):
            file_list.append(os.path.join(data_dir, files))

    #print(file_list)
    code_dict = {}
    for f in file_list:
        fp = open(f)
        temp = []
        for i in fp.readlines():
            temp.append(list(i))
        dat = [item for sublist in temp for item in sublist]
        code_dict[f] = dat
    #print(code_dict)
    return code_dict

def get_word_embeddings(data):
    # Need to generate a word embeddings for the text which can be as simple as the assigned numbers

    res = []
    for i in data:
        res.append(vocab[i])

    return res
'''
data = get_text('../data')
for i in data.keys():
    embeddings = get_word_embeddings(data[i])
    print(embeddings)
'''



