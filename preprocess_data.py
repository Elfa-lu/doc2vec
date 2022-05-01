"""
credit to: https://github.com/yoonkim/CNN_sentence
pre-trained word vectors downloaded from:
    https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
Google word2vec: https://code.google.com/archive/p/word2vec/

"""

import numpy as np
import _pickle as cPickle
from collections import defaultdict
import re
import pandas as pd
import os
import io


def build_data_cv(data_folder, cv=10, clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    pos_file = data_folder[0]
    neg_file = data_folder[1]
    vocab = defaultdict(float)

    with open(pos_file, "r", errors="ignore") as f:
        for line in f:
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            if "rt" in pos_file:
                datum = {
                    "y": 1,
                    "text": orig_rev,
                    "num_words": len(orig_rev.split()),
                    "split": np.random.randint(0, cv),
                }
            else:
                datum = {
                    "y": 1,
                    "text": orig_rev[2:],
                    "num_words": len(orig_rev.split()),
                    "split": np.random.randint(0, cv),
                }
            revs.append(datum)

    with open(neg_file, "r", errors="ignore") as f:
        for line in f:
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            if "rt" in pos_file:
                datum = {
                    "y": 0,
                    "text": orig_rev,
                    "num_words": len(orig_rev.split()),
                    "split": np.random.randint(0, cv),
                }
            else:
                datum = {
                    "y": 0,
                    "text": orig_rev[2:],
                    "num_words": len(orig_rev.split()),
                    "split": np.random.randint(0, cv),
                }
            revs.append(datum)

    return revs, vocab


def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size + 1, k), dtype="float32")
    W[0] = np.zeros(k, dtype="float32")
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def loan_bin_vec_fasttext_wiki(fname, vocab):
    """
    Loads 300x1 word vecs from FastText wiki-news-300d-1M.vec.zip
    """
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        word = tokens[0]
        if word in vocab:
            data[word] = np.array(tokens[1:], dtype=np.float64)
    return data


def loan_bin_vec_fasttext_crawl(fname, vocab):
    """
    Loads 300x1 word vecs from FastText crawl-300d-2M.vec.zip
    """
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        word = tokens[0]
        if word in vocab:
            data[word] = np.array(tokens[1:], dtype=np.float64)
    return data


def loan_bin_vec_glove(fname, vocab):
    """
    Loads 300x1 word vecs from Stanford GloVe
    """
    word_vecs = {}
    with open(fname, "r") as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            if word in vocab:
                embedding = np.array(split_line[-300:], dtype=np.float64)
                word_vecs[word] = embedding
    return word_vecs


def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype("float32").itemsize * layer1_size
        for _ in range(vocab_size):
            word = []
            while True:
                # The file is not utf-8 encoded, it is ISO-8859-1 encoded
                ch = f.read(1).decode("ISO-8859-1")
                if ch == " ":
                    word = "".join(word)
                    break
                if ch != "\n":
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype="float32")
            else:
                f.read(binary_len)
    return word_vecs


def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " 's", string)
    string = re.sub(r"\'ve", " 've", string)
    string = re.sub(r"n\'t", " n't", string)
    string = re.sub(r"\'re", " 're", string)
    string = re.sub(r"\'d", " 'd", string)
    string = re.sub(r"\'ll", " 'll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


if __name__ == "__main__":
    cur_dir = os.getcwd()
    parent_dir = os.path.dirname(cur_dir)

    datasets = ["rt", "mpqa", "cr", "subj"]
    for dataset in datasets:
        file_name_pos = dataset + ".pos"
        file_name_neg = dataset + '.neg'
        data_folder = [
            os.path.join(parent_dir, "datasets", dataset, file_name_pos),
            os.path.join(parent_dir, "datasets", dataset, file_name_neg),
        ]

        # load word2vec 300d
        w2v_file = os.path.join(parent_dir, "GoogleNews-vectors-negative300.bin")
        print("loading data...")

        revs, vocab = build_data_cv(data_folder, cv=10, clean_string=True)
        max_l = np.max(pd.DataFrame(revs)["num_words"])
        print("data loaded!")
        print("number of sentences: " + str(len(revs)))
        print("vocab size: " + str(len(vocab)))
        print("max sentence length: " + str(max_l))

        print("loading word2vec vectors...")
        w2v = load_bin_vec(w2v_file, vocab)
        print("word2vec loaded!")
        print("num words already in word2vec: " + str(len(w2v)))
        add_unknown_words(w2v, vocab)
        W, word_idx_map = get_W(w2v)

        # load glove 300d
        # Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors)
        glove_file = os.path.join(parent_dir, "glove.840B.300d.txt")
        print("loading glove vectors...")
        glove = loan_bin_vec_glove(glove_file, vocab)
        print("glove loaded!")
        print("num words already in glove: " + str(len(glove)))
        add_unknown_words(glove, vocab)
        W_glove, word_idx_map_glove = get_W(glove)

        # load fasttext wiki
        ft_wiki_file = os.path.join(parent_dir, "wiki-news-300d-1M.vec")
        print("loading fasttext wiki vectors...")
        ft_wiki = loan_bin_vec_fasttext_wiki(ft_wiki_file, vocab)
        print("fasttext wiki loaded!")
        print("num words already in fasttext wiki: " + str(len(ft_wiki)))
        add_unknown_words(ft_wiki, vocab)
        W_ft_wiki, word_idx_map_ft_wiki = get_W(ft_wiki)

        # load fasttext crawl
        ft_crawl_file = os.path.join(parent_dir, "crawl-300d-2M.vec")
        print("loading fasttext crawl vectors...")
        ft_crawl = loan_bin_vec_glove(ft_crawl_file, vocab)
        print("fasttext crawl loaded!")
        print("num words already in fasttext crawl: " + str(len(ft_crawl)))
        add_unknown_words(ft_crawl, vocab)
        W_ft_crawl, word_idx_map_ft_crawl = get_W(ft_crawl)

        rand_vecs = {}
        add_unknown_words(rand_vecs, vocab)
        W2, _ = get_W(rand_vecs)

        file_name = dataset + ".p"
        cPickle.dump(
            [
                revs, W, W2, word_idx_map, vocab, 
                W_glove, word_idx_map_glove, 
                W_ft_wiki, word_idx_map_ft_wiki, 
                W_ft_crawl, word_idx_map_ft_crawl
            ],
            open(file_name, "wb"),
        )
        print("dataset created!")
