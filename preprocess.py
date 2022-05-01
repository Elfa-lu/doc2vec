import numpy as np
import h5py
import re
import operator
import argparse
import io


################## FOR DATASET WITH DEV/TEST SET ##################

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


def line_to_words(line, dataset):
    if dataset.upper() == "SST1" or dataset.upper() == "SST2":
        clean_line = clean_str_sst(line.strip())
    else:
        clean_line = clean_str(line.strip())
    words = clean_line.split(" ")
    words = words[1:]

    return words


def get_vocab(file_list, dataset=""):
    max_sent_len = 0
    word_to_idx = {}
    # Starts at 2 for padding
    idx = 2

    for filename in file_list:
        f = open(filename, "r")
        for line in f:
            words = line_to_words(line, dataset)
            max_sent_len = max(max_sent_len, len(words))
            for word in words:
                if not word in word_to_idx:
                    word_to_idx[word] = idx
                    idx += 1

        f.close()
    return max_sent_len, word_to_idx


def load_data(dataset, train_name, test_name="", dev_name="", padding=4):
    """
    Load training data (dev/test optional).
    """
    f_names = [train_name]
    if not test_name == "":
        f_names.append(test_name)
    if not dev_name == "":
        f_names.append(dev_name)

    max_sent_len, word_to_idx = get_vocab(f_names, dataset)

    dev = []
    dev_label = []
    train = []
    train_label = []
    test = []
    test_label = []

    files = []
    data = []
    data_label = []

    f_train = open(train_name, "r")
    files.append(f_train)
    data.append(train)
    data_label.append(train_label)
    if not test_name == "":
        f_test = open(test_name, "r")
        files.append(f_test)
        data.append(test)
        data_label.append(test_label)
    if not dev_name == "":
        f_dev = open(dev_name, "r")
        files.append(f_dev)
        data.append(dev)
        data_label.append(dev_label)

    for d, lbl, f in zip(data, data_label, files):
        for line in f:
            words = line_to_words(line, dataset)
            y = int(line.strip().split()[0]) + 1
            sent = [word_to_idx[word] for word in words]
            # end padding
            if len(sent) < max_sent_len + padding:
                sent.extend([1] * (max_sent_len + padding - len(sent)))
            # start padding
            sent = [1] * padding + sent

            d.append(sent)
            lbl.append(y)

    f_train.close()
    if not test_name == "":
        f_test.close()
    if not dev_name == "":
        f_dev.close()

    return (
        word_to_idx,
        np.array(train, dtype=np.int32),
        np.array(train_label, dtype=np.int32),
        np.array(test, dtype=np.int32),
        np.array(test_label, dtype=np.int32),
        np.array(dev, dtype=np.int32),
        np.array(dev_label, dtype=np.int32),
    )


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
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
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def clean_str_sst(string):
    """
  Tokenization/string cleaning for the SST dataset
  """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


FILE_PATHS = {
    "SST1": (
        "../datasets/stsa-1/stsa.fine.phrases.train",
        "../datasets/stsa-1/stsa.fine.dev",
        "../datasets/stsa-1/stsa.fine.test",
    ),
    "SST2": (
        "../datasets/stsa-2/stsa.binary.phrases.train",
        "../datasets/stsa-2/stsa.binary.dev",
        "../datasets/stsa-2/stsa.binary.test",
    ),
    "MR": ("data/rt-polarity.all", "", ""),
    "SUBJ": ("data/subj.all", "", ""),
    "CR": ("data/custrev.all", "", ""),
    "MPQA": ("data/mpqa.all", "", ""),
    "TREC": ("../datasets/trec/TREC.train.all.txt", "", "../datasets/trec/TREC.test.all.txt"),
}
args = {}


def main():
    global args
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset", help="Data set", type=str)
    parser.add_argument("w2v", help="word2vec file", type=str)
    parser.add_argument("glove", help="glove file", type=str)
    parser.add_argument("fasttext_wiki", help="fasttext wiki file", type=str)
    parser.add_argument("fasttext_crawl", help="fasttext crawl file", type=str)
    parser.add_argument("--train", help="custom train data", type=str, default="")
    parser.add_argument("--test", help="custom test data", type=str, default="")
    parser.add_argument("--dev", help="custom dev data", type=str, default="")
    parser.add_argument(
        "--padding", help="padding around each sentence", type=int, default=4
    )
    parser.add_argument(
        "--custom_name",
        help="name of custom output hdf5 file",
        type=str,
        default="custom",
    )
    args = parser.parse_args()
    dataset = args.dataset

    # Dataset name
    if dataset == "custom":
        # Train on custom dataset
        train_path, dev_path, test_path = args.train, args.dev, args.test
        dataset = args.custom_name
    else:
        train_path, dev_path, test_path = FILE_PATHS[dataset]

    # Load data
    word_to_idx, train, train_label, test, test_label, dev, dev_label = load_data(
        dataset,
        train_path,
        test_name=test_path,
        dev_name=dev_path,
        padding=args.padding,
    )

    # Write word mapping to text file.
    with open(dataset + "_word_mapping.txt", "w+") as embeddings_f:
        embeddings_f.write("*PADDING* 1\n")
        for word, idx in sorted(word_to_idx.items(), key=operator.itemgetter(1)):
            embeddings_f.write("%s %d\n" % (word, idx))

    # Load word2vec
    w2v = load_bin_vec(args.w2v, word_to_idx)
    V = len(word_to_idx) + 1
    print("Vocab size:", V)

    # Not all words in word_to_idx are in w2v.
    # Word embeddings initialized to random Unif(-0.25, 0.25)
    embed = np.random.uniform(-0.25, 0.25, (V, len(list(w2v.values())[0])))
    embed[0] = 0
    for word, vec in w2v.items():
        embed[word_to_idx[word] - 1] = vec

    # Load glove
    glove = loan_bin_vec_glove(args.glove, word_to_idx)
    embed_glove = np.random.uniform(-0.25, 0.25, (V, len(list(glove.values())[0])))
    embed_glove[0] = 0
    for word, vec in glove.items():
        embed_glove[word_to_idx[word] - 1] = vec

    # Load fasttext wiki
    fasttext_wiki = loan_bin_vec_fasttext_wiki(args.fasttext_wiki, word_to_idx)
    embed_ft_wiki = np.random.uniform(-0.25, 0.25, (V, len(list(fasttext_wiki.values())[0])))
    embed_ft_wiki[0] = 0
    for word, vec in fasttext_wiki.items():
        embed_ft_wiki[word_to_idx[word] - 1] = vec

    # Load fasttext crawl
    fasttext_crawl = loan_bin_vec_fasttext_crawl(args.fasttext_crawl, word_to_idx)
    embed_ft_crawl = np.random.uniform(-0.25, 0.25, (V, len(list(fasttext_crawl.values())[0])))
    embed_ft_crawl[0] = 0
    for word, vec in fasttext_crawl.items():
        embed_ft_crawl[word_to_idx[word] - 1] = vec

    print("train size:", train.shape)

    filename = dataset + ".hdf5"
    with h5py.File(filename, "w") as f:
        f["w2v"] = np.array(embed)
        f["glove"] = np.array(embed_glove)
        f["ft_wiki"] = np.array(embed_ft_wiki)
        f["ft_crawl"] = np.array(embed_ft_crawl)
        f["train"] = train
        f["train_label"] = train_label
        f["test"] = test
        f["test_label"] = test_label
        f["dev"] = dev
        f["dev_label"] = dev_label


if __name__ == "__main__":
    main()
    # python preprocess.py SST1 ../GoogleNews-vectors-negative300.bin ../glove.840B.300d.txt ../wiki-news-300d-1M.vec ../crawl-300d-2M.vec
    # python preprocess.py SST2 ../GoogleNews-vectors-negative300.bin ../glove.840B.300d.txt ../wiki-news-300d-1M.vec ../crawl-300d-2M.vec
    # python preprocess.py TREC ../GoogleNews-vectors-negative300.bin ../glove.840B.300d.txt ../wiki-news-300d-1M.vec ../crawl-300d-2M.vec
    