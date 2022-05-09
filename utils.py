from collections import defaultdict
import _pickle as cPickle
import numpy as np
import h5py


def load_preprocessed_data_sst(dataset="sst1", k=300):
    file_name = dataset.upper() + ".hdf5"
    f = h5py.File(file_name, "r")

    dev = np.array(f["dev"])
    dev_label = np.array(f["dev_label"])
    test = np.array(f["test"])
    test_label = np.array(f["test_label"])
    train = np.array(f["train"])
    train_label = np.array(f["train_label"])
    W = np.array(f["w2v"])
    W_glove = np.array(f["glove"])
    W_ft_wiki = np.array(f["ft_wiki"])
    W_ft_crawl = np.array(f["ft_crawl"])

    if dataset == "trec":
        data = np.concatenate((train, test), axis=0)
        data_label = np.concatenate((train_label, test_label), axis=0)
    else:
        data = np.concatenate((train, dev, test), axis=0)
        data_label = np.concatenate((train_label, dev_label, test_label), axis=0)

    revs = []

    vocab = defaultdict(float)
    vocab["*PADDING*"] = 1

    word_idx_map = {}
    word_idx_map_reverse = {}
    word_mapping_file = dataset.upper() + "_word_mapping.txt"
    with open(word_mapping_file, "r") as fileobj:
        for row in fileobj:
            lst = row.rstrip("\n").split(" ")
            word_idx_map[lst[0]] = int(lst[1])
            word_idx_map_reverse[int(lst[1])] = lst[0]

    for i in range(data.shape[0]):
        if all(data[i] == 1):
            continue
        dict = {}
        dict["y"] = data_label[i]
        start_idx = np.where(data[i] != 1)[0][0]
        last_idx = np.where(data[i] != 1)[0][-1]
        words_lst = []
        for idx in data[i][start_idx : last_idx + 1]:
            words_lst.append(word_idx_map_reverse[idx])
            vocab[word_idx_map_reverse[idx]] += 1

        dict["text"] = " ".join(words_lst)
        dict["num_words"] = len(data[i]) - list(data[i]).count(1)
        dict["split"] = 0
        revs.append(dict)

    vocab_size = W.shape[0]

    W2 = np.zeros(shape=(vocab_size + 1, k), dtype="float32")
    W2[0] = np.zeros(k, dtype="float32")
    for i in range(1, vocab_size + 1):
        W2[i] = np.random.uniform(-0.25, 0.25, k)

    return revs, W, W2, word_idx_map, vocab, W_glove, word_idx_map, W_ft_wiki, word_idx_map, W_ft_crawl, word_idx_map


def load_preprocessed_data(dataset="rt"):
    """
    revs : list of dict
        each item in the list contains target(y), text, numwords(num_words), and split
    W : (18778, 300) 2D array
        word embedding numpy array
    word_idx_map : dict
        word - index map for W
    vocab : defaultdict
        word counts, len() = 18778
    """
    filename = dataset.lower() + ".p"
    revs, W, W2, word_idx_map, vocab, W_glove, word_idx_map_glove, W_ft_wiki, word_idx_map_ft_wiki, W_ft_crawl, word_idx_map_ft_crawl = cPickle.load(
        open(filename, "rb")
    )
    return revs, W, W2, word_idx_map, vocab, W_glove, word_idx_map_glove, W_ft_wiki, word_idx_map_ft_wiki, W_ft_crawl, word_idx_map_ft_crawl


def get_doc_vec_options(dataset, revs, W, word_idx_map, option="mean"):
    """
    Returns the mean/max/min embedding vector for each sentence. len(revs)*len(vocab)
    """
    doc_embedding = []
    for sentence in revs:
        text = sentence["text"]
        numwords = sentence["num_words"]

        if dataset == "sst1" or dataset == "sst2" or dataset == "trec":
            if len(text) > 0:
                embedding = [W[word_idx_map[word] - 1] for word in text.split(" ")]
        else:
            if len(text) > 0:
                embedding = [W[word_idx_map[word]] for word in text.split(" ")]

        if option == "mean":
            doc_embedding.append(sum(embedding) / numwords)
        elif option == "max":
            # a = np.array(embedding).max(axis=0) if numwords > 1 else np.reshape(embedding, (-1,))
            # if a.shape != (300,):
            #     print(a.shape)
            doc_embedding.append(
                np.array(embedding).max(axis=0)
                if numwords > 1
                else np.reshape(embedding, (-1,))
            )
        elif option == "min":
            doc_embedding.append(
                np.array(embedding).min(axis=0)
                if numwords > 1
                else np.reshape(embedding, (-1,))
            )  # list(map(min, *embedding))

    return doc_embedding


def get_doc_vec_multi(dataset, revs, W_1, word_idx_map_1, W_2, word_idx_map_2, W_3=None, word_idx_map_3=None):
    doc_embedding = []

    for sentence in revs:
        text = sentence["text"]
        numwords = sentence["num_words"]

        # if len(text) == 0 and W_3 is None:
        #     doc_embedding.append(np.random.uniform(-0.25, 0.25, 900))
        #     continue

        # if len(text) == 0 and not W_3 is None:
        #     doc_embedding.append(np.random.uniform(-0.25, 0.25, 600))
        #     continue
        
        if dataset == "sst1" or dataset == "sst2" or dataset == "trec":
            if len(text) > 0:
                embedding_1 = [W_1[word_idx_map_1[word] - 1] for word in text.split(" ")]
                embedding_2 = [W_2[word_idx_map_2[word] - 1] for word in text.split(" ")]
                if W_3 is not None:
                    embedding_3 = [W_3[word_idx_map_3[word] - 1] for word in text.split(" ")]
        else:
            if len(text) > 0:
                embedding_1 = [W_1[word_idx_map_1[word]] for word in text.split(" ")]
                embedding_2 = [W_2[word_idx_map_2[word]] for word in text.split(" ")]
                if W_3 is not None:
                    embedding_3 = [W_3[word_idx_map_3[word]] for word in text.split(" ")]

        embedding_1_ = sum(embedding_1) / numwords
        embedding_2_ = sum(embedding_2) / numwords
        if W_3 is None:
            embedding = [*embedding_1_, *embedding_2_]
        else:
            embedding_3_ = sum(embedding_3) / numwords
            embedding = [*embedding_1_, *embedding_2_, *embedding_3_]
            
        doc_embedding.append(embedding)

    return doc_embedding


def save_embeddings(dataset, revs, W_1, word_idx_map_1, W_2, word_idx_map_2, W_3, word_idx_map_3):
    embeddings = []

    for sentence in revs:
        text = sentence["text"]
        target = sentence["y"]
        
        if len(text) == 0:
            continue

        if dataset == "sst1" or dataset == "sst2" or dataset == "trec":
            embedding_1 = [W_1[word_idx_map_1[word] - 1] for word in text.split(" ")]
            embedding_2 = [W_2[word_idx_map_2[word] - 1] for word in text.split(" ")]
            embedding_3 = [W_3[word_idx_map_3[word] - 1] for word in text.split(" ")]
        else:
            embedding_1 = [W_1[word_idx_map_1[word]] for word in text.split(" ")]
            embedding_2 = [W_2[word_idx_map_2[word]] for word in text.split(" ")]
            embedding_3 = [W_3[word_idx_map_3[word]] for word in text.split(" ")]

        embeddings.append([embedding_1, embedding_2, embedding_3, target])
    
    filename = "w2v_glo_ft_embeddings_" + dataset
    cPickle.dump(embeddings, open(filename, "wb"))


def get_doc_vec_bow(revs, word_idx_map, vocab):
    """
    Returns the bow vector for each sentence. len(revs)*len(vocab)
    """
    vocab_size = len(vocab)
    doc_bow = []
    for sentence in revs:
        sentence_bow = np.zeros(vocab_size)
        text = sentence["text"]
        words = text.split(" ")
        for word in words:
            if len(word) > 0:
                sentence_bow[word_idx_map[word] - 1] += 1
        doc_bow.append(sentence_bow)
    return doc_bow


def idf(revs, word_idx_map, vocab):
    # log of the ratio of the number of documents to the number of documents containing the word.
    # We take log of this ratio because when the corpus becomes large IDF values
    # can get large causing it to explode hence taking log will dampen this effect.
    # we cannot divide by 0, we smoothen the value by adding 1 to the denominator.
    word_count = defaultdict(int)
    total_sentences = len(revs)
    total_words = len(vocab)
    idf = np.array([0.0 for _ in range(total_words)])

    # document frequency for each word
    for word in vocab:
        for sentence in revs:
            text = sentence["text"]
            if word in text:
                word_count[word] += 1

    # calculate idf
    for word in vocab:
        if word_count[word] == 0:
            word_count[word] += 1
        idf[word_idx_map[word] - 1] = np.log(total_sentences / word_count[word])

    return idf


def tf(sentence, word_idx_map, vocab, num_of_word):
    text = sentence["text"]
    words = text.split(" ")

    N = len(words)
    tf = np.array([0.0 for _ in range(len(vocab))])
    for word in words:
        if len(word) > 0:
            num_of_word[word] += 1
    for word, cnt in num_of_word.items():
        tf[word_idx_map[word] - 1] = cnt / float(N)

    return tf


def get_doc_vec_tfidf(revs, word_idx_map, vocab):
    """
    Returns the tfidf vector for each sentence. len(revs)*len(vocab)
    """
    word_set = set(vocab)
    num_of_word = dict.fromkeys(word_set, 0)

    idf_ = idf(revs, word_idx_map, vocab)
    doc_tfidf = []

    for sentence in revs:
        tf_ = tf(sentence, word_idx_map, vocab, num_of_word)
        tfidf = [a * b for a, b in zip(idf_, tf_)]
        doc_tfidf.append(tfidf)

    return doc_tfidf


def lsa(revs, num_components=2, random_state=0):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    doc = []
    for rev in revs:
        doc.append(rev["text"])

    vectorizer = TfidfVectorizer(stop_words="english")
    X_tfidf = vectorizer.fit_transform(doc)

    lsa = TruncatedSVD(n_components=num_components, random_state=random_state)
    lsa.fit_transform(X_tfidf)

    words = vectorizer.get_feature_names()

    one_hot_encoding_words = set()
    for index, component in enumerate(lsa.components_):
        zipped = zip(words, component)
        top_terms_key = sorted(zipped, key=lambda t: t[1], reverse=True)[:10]
        top_terms_list = list(dict(top_terms_key).keys())
        print("Topic " + str(index) + ": ", top_terms_list)
        one_hot_encoding_words.update(top_terms_list)

    return list(one_hot_encoding_words)


def add_lsa_features(revs):
    one_hot_encoding_words = lsa(revs)

    features = []
    for rev in revs:
        feature = []
        sentence = rev["text"]
        for word in one_hot_encoding_words:
            feature.append(int(word in sentence))
        features.append(feature)

    return features


if __name__ == "__main__":
    revs, W, W2, word_idx_map, vocab, W_glove, word_idx_map_glove, W_ft_wiki, word_idx_map_ft_wiki, W_ft_crawl, word_idx_map_ft_crawl = load_preprocessed_data("cr")
    save_embeddings("cr", revs, W, word_idx_map, W_glove, word_idx_map_glove, W_ft_crawl, word_idx_map_ft_crawl)

    # features = add_lsa_features(revs)
