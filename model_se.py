from collections import defaultdict
from utils import *
import h5py
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import random
import time


def train_model(dataset, algorithm="se", cv=10, random_state=0):
    revs, W,  _, word_idx_map, vocab, W_glove, word_idx_map_glove = load_preprocessed_data(dataset)
    results = []

    for i in range(cv):
        train_, test = [], []
        for rev in revs:
            if rev["split"] == i:            
                test.append(rev)        
            else:  
                train_.append(rev)   

        train_size = len(train_)
        train = train_[:int(train_size * 0.9)]
        dev = train_[int(train_size * 0.9):]

        X_train_, hx_ = get_doc_vec_multi_info(dataset, algorithm, train_, W, word_idx_map, W_glove, word_idx_map_glove)
        X_train, hx = get_doc_vec_multi_info(dataset, algorithm, train, W, word_idx_map, W_glove, word_idx_map_glove)
        X_dev = get_doc_vec_multi_info_test(dataset, algorithm, dev, hx, W, word_idx_map, W_glove, word_idx_map_glove)
        X_test = get_doc_vec_multi_info_test(dataset, algorithm, test, hx_, W, word_idx_map, W_glove, word_idx_map_glove)
        
        y_train_ = [rev["y"] for rev in train_]
        y_train = [rev["y"] for rev in train]
        y_dev = [rev["y"] for rev in dev]
        y_test = [rev["y"] for rev in test]  

        X_train_ = pd.DataFrame(X_train_)
        X_train = pd.DataFrame(X_train)
        X_dev = pd.DataFrame(X_dev)
        X_test = pd.DataFrame(X_test)
        y_train_ = pd.Series(y_train_)
        y_train = pd.Series(y_train)
        y_dev = pd.Series(y_dev)
        y_test = pd.Series(y_test)

        best_acc = 0

        kernel = "rbf"
        gammas = [1e-1, 1e-2, 1e-3, 1e-4]
        Cs = [3, 10, 100, 1000]
        for gamma in gammas:
            for c in Cs:
                start_time = time.time()
                clf = SVC(
                    kernel=kernel,
                    gamma=gamma,
                    C=c,
                    class_weight="balanced",
                    random_state=random_state,
                )
                clf.fit(X_train, y_train)
                acc = accuracy_score(y_dev, clf.predict(X_dev))
                print("--- %s seconds ---" % (time.time() - start_time))
                print(
                    "ker:" + str(kernel) + "  gamma:" + str(gamma) + "  c:" + str(c),
                    flush=True,
                )
                print("acc:" + str(acc), flush=True)
                print()

                if acc > best_acc:
                    best_acc = acc
                    best_clf = clf

        kernel = "poly"
        degrees = [i for i in range(1, 5)]
        for degree in degrees:
            start_time = time.time()
            clf = SVC(
                kernel=kernel,
                degree=degree,
                class_weight="balanced",
                random_state=random_state,
            )
            clf.fit(X_train, y_train)
            acc = accuracy_score(y_dev, clf.predict(X_dev))
            print("--- %s seconds ---" % (time.time() - start_time))
            print("ker:" + str(kernel) + "  degree:" + str(degree), flush=True)
            print("acc:" + str(acc), flush=True)
            print()

            if acc > best_acc:
                best_acc = acc
                best_clf = clf

        best_clf.fit(X_train_, y_train_)
        acc = accuracy_score(y_test, best_clf.predict(X_test))
        print("cv: " + str(i + 1) + ", acc: " + str(acc), flush=True)
        results.append(acc)

    print("accuracy for each fold: " + str(results), flush=True)
    print("10 fold vc validation result for SVM: ", str(np.mean(results)), flush=True)


def train_model_has_dev_set(dataset, algorithm="se", random_state=0, cv=10):
    revs, W,  _, word_idx_map, vocab, W_glove, word_idx_map_glove = load_preprocessed_data_sst(dataset)

    if dataset == "sst1" or dataset == "sst2":
        file_name = dataset.upper() + ".hdf5"
        f = h5py.File(file_name, "r")
        train_size = f["train_label"].shape[0]
        dev_size = f["dev_label"].shape[0]

        X_train_, hx_ = get_doc_vec_multi_info(dataset, algorithm, revs[:train_size + dev_size], W, word_idx_map, W_glove, word_idx_map_glove)
        X_train, hx = get_doc_vec_multi_info(dataset, algorithm, revs[:train_size], W, word_idx_map, W_glove, word_idx_map_glove)
        X_dev = get_doc_vec_multi_info_test(dataset, algorithm, revs[train_size : train_size + dev_size], hx, W, word_idx_map, W_glove, word_idx_map_glove)
        X_test = get_doc_vec_multi_info_test(dataset, algorithm, revs[train_size + dev_size :], hx, W, word_idx_map, W_glove, word_idx_map_glove)
        
        target = [rev["y"] for rev in revs]
        y_train_ = target[:train_size + dev_size]
        y_train = target[:train_size]
        y_dev = target[train_size : train_size + dev_size]
        y_test = target[train_size + dev_size :]

    elif dataset == "trec":
        file_name = dataset.upper() + ".hdf5"
        f = h5py.File(file_name, "r")
        train_size = f["train_label"].shape[0]
        dev_size = f["test_label"].shape[0]

        random.seed(random_state)
        idx = [*range(train_size)]
        random.shuffle(idx)
        idx_train = idx[:int(train_size*0.9)]
        idx_dev = idx[int(train_size*0.9):]
        revs_train = [element for i, element in enumerate(revs) if i in idx_train]
        revs_dev = [element for i, element in enumerate(revs) if i in idx_dev]

        X_train_, hx_ = get_doc_vec_multi_info(dataset, algorithm, revs_train + revs_dev, W, word_idx_map, W_glove, word_idx_map_glove)
        X_train, hx = get_doc_vec_multi_info(dataset, algorithm, revs_train, W, word_idx_map, W_glove, word_idx_map_glove)
        X_dev = get_doc_vec_multi_info_test(dataset, algorithm, revs_dev, hx, W, word_idx_map, W_glove, word_idx_map_glove)
        X_test = get_doc_vec_multi_info_test(dataset, algorithm, revs[train_size:], hx_, W, word_idx_map, W_glove, word_idx_map_glove)
        
        target = [rev["y"] for rev in revs]
        y_train = [element for i, element in enumerate(target) if i in idx_train]
        y_dev = [element for i, element in enumerate(target) if i in idx_dev]
        y_train_ = y_train + y_dev
        y_test = target[train_size:]

    train_size = len(y_train)
    dev_size = len(y_dev)

    X_train = pd.DataFrame(X_train)
    X_dev = pd.DataFrame(X_dev)
    X_test = pd.DataFrame(X_test)
    y_train = pd.Series(y_train)
    y_dev = pd.Series(y_dev)
    y_test = pd.Series(y_test)

    best_acc = 0

    kernel = "rbf"
    gammas = [1e-1, 1e-2, 1e-3, 1e-4]
    Cs = [3, 10, 100, 1000]
    for gamma in gammas:
        for c in Cs:
            start_time = time.time()
            clf = SVC(
                kernel=kernel,
                gamma=gamma,
                C=c,
                class_weight="balanced",
                random_state=random_state,
            )
            clf.fit(X_train, y_train)

            acc = accuracy_score(y_dev, clf.predict(X_dev))
            print("--- %s seconds ---" % (time.time() - start_time))
            print(
                "ker:" + str(kernel) + "  gamma:" + str(gamma) + "  c:" + str(c),
                flush=True,
            )
            print("acc:" + str(acc))
            print()

            if acc > best_acc:
                best_acc = acc
                best_clf = clf

    kernel = "poly"
    degrees = [i for i in range(1, 5)]
    for degree in degrees:
        start_time = time.time()
        clf = SVC(
            kernel=kernel,
            degree=degree,
            class_weight="balanced",
            random_state=random_state,
        )
        clf.fit(X_train, y_train)

        acc = accuracy_score(y_dev, clf.predict(X_dev))
        print("--- %s seconds ---" % (time.time() - start_time))
        print("ker:" + str(kernel) + "  degree:" + str(degree), flush=True)
        print("acc:" + str(acc))
        print()

        if acc > best_acc:
            best_acc = acc
            best_clf = clf

    print("best_acc:" + str(best_acc) + "  optimal_clf:" + str(best_clf))

    # accuracy on test set using the best parameter
    best_clf.fit(X_train_, y_train_)
    pred_val = best_clf.predict(X_test)
    acc = accuracy_score(y_test, pred_val)
    print("accuracy on test dataset: " + str(acc))


################### Calculate SE and apply weights to training set ###################
def get_train_vocab(revs):
    vocab = defaultdict(int)

    for sentence in revs: 
        text = sentence["text"]
        words = text.split(" ")
        for word in set(words):
            vocab[word] += 1
    
    return vocab


def shannon_entropy_train(revs):
    vocab = get_train_vocab(revs)
    hx = dict.fromkeys(vocab, 0)
    
    revs_df = create_revs_df(revs)
    # categories = revs_df["y"].unique()

    for x_value in vocab:
        x_value_idx = []
        for index, row in revs_df.iterrows():
            text = row['sentence']
            if x_value in text.split(" "):
                x_value_idx.append(index)

        revs_x_df = revs_df.filter(items=x_value_idx, axis=0)
        probs = revs_x_df["y"].value_counts(normalize=True)
        se = -1 * np.sum(np.log2(probs) * probs)
        hx[x_value] = se
        
    # vocab = get_train_vocab(revs)
    # dict_neg = dict.fromkeys(vocab, 0)

    # for sentence in revs:
    #     text = sentence["text"]
    #     words = text.split(" ")
    #     label = sentence["y"]
    #     if label:
    #         for word in set(words):
    #             dict_neg[word] += 1

    # prob_dict_neg = {k: dict_neg[k] / vocab[k] for k in vocab}

    # hx = defaultdict(float)
    # for k in vocab:
    #     if prob_dict_neg[k] == 1 or prob_dict_neg[k] == 0:
    #         # hx[k] = 2.5
    #         hx[k] = 0
    #     else:
    #         hx[k] = -prob_dict_neg[k] * (math.log(prob_dict_neg[k], 2)) - (1 - prob_dict_neg[k]) * (math.log((1 - prob_dict_neg[k]), 2))

    # hx = {k: abs(hx[k] - 1) + 1 for k in hx}
    hx = {k: abs(hx[k] - 1) for k in hx}

    return hx


def get_doc_vec_multi_info(dataset, algorithm, revs, W, word_idx_map, W_glove, word_idx_map_glove):
    doc_embedding = []

    if algorithm == "se":
        hx = shannon_entropy_train(revs)
    elif algorithm == "ig":
        hx = information_gain_train(revs)

    for sentence in revs:
        text = sentence["text"]

        if dataset == "sst1" or dataset == "sst2" or dataset == "trec":
            if len(text) > 0:
                hx_weights = sum([hx[word] for word in text.split(" ")])
                embedding_w2v = [W[word_idx_map[word] - 1] * hx[word] for word in text.split(" ")]
                embedding_glove = [W_glove[word_idx_map_glove[word] - 1] * hx[word] for word in text.split(" ")]
        else:
            if len(text) > 0:
                hx_weights = sum([hx[word] for word in text.split(" ")])
                embedding_w2v = [W[word_idx_map[word]] * hx[word] for word in text.split(" ")]
                embedding_glove = [W_glove[word_idx_map_glove[word]] * hx[word] for word in text.split(" ")]

        # print([hx[word] for word in text.split(" ")])
        # for those got prob(x|true) == prob(x|false) == 1/2, se == 0, sum(hx_weights) is 0
        if hx_weights == 0:
            doc_embedding.append(doc_embedding[-1])
            continue

        embedding_w2v_ = sum(embedding_w2v) / hx_weights
        embedding_glove_ = sum(embedding_glove) / hx_weights
        embedding = [*embedding_w2v_, *embedding_glove_]
        doc_embedding.append(embedding)
    
    return doc_embedding, hx


################### Calculate SE and apply weights to test set ###################
def calculate_test_hx(hx_weights):
    if 999 not in hx_weights:
        return hx_weights

    weight_sum = 0
    cnt = 0
    for hx_weight in hx_weights:
        if hx_weight == 999:
            continue
        weight_sum += hx_weight
        cnt += 1

    # all items in list is unknown
    if cnt == 0:
        return [1 for _ in hx_weights]

    fill_weight = weight_sum / cnt

    # for vocabs not appear in training set
    # fill the weights by using the average sentence weights
    for idx, weight in enumerate(hx_weights):
        if weight == 999:
            hx_weights[idx] = fill_weight
    
    return hx_weights
            

def get_doc_vec_multi_info_test(dataset, algorithm, revs, hx, W, word_idx_map, W_glove, word_idx_map_glove):
    doc_embedding = []

    # if algorithm == "se":
    #     hx = shannon_entropy_train(revs)
    # elif algorithm == "ig":
    #     hx = information_gain_train(revs)

    for sentence in revs:
        text = sentence["text"]

        if len(text) == 0:
            doc_embedding.append(doc_embedding[-1])
            continue
        
        hx_weights= []
        for word in text.split(" "):
            if word in hx:
                hx_weights.append(hx[word])
            else:
                hx_weights.append(999)
        
        hx_weights = calculate_test_hx(hx_weights)

        if dataset == "sst1" or dataset == "sst2" or dataset == "trec":
            embedding_w2v = [W[word_idx_map[word] - 1] for word in text.split(" ")]
            embedding_w2v_hx = [np.array(embedding_w2v[i]) * np.array(hx_weights[i]) for i in range(len(embedding_w2v))]
            # embedding_w2v_hx = [a * b for a, b in zip(hx_weights, embedding_w2v)]
            embedding_glove = [W_glove[word_idx_map_glove[word] - 1] for word in text.split(" ")]
            embedding_glove_hx = [np.array(embedding_glove[i]) * np.array(hx_weights[i]) for i in range(len(embedding_glove))]
            # embedding_glove_hx = [a * b for a, b in zip(hx_weights, embedding_glove)]
        
        else:
            embedding_w2v = [W[word_idx_map[word]] for word in text.split(" ")]
            embedding_w2v_hx = [np.array(embedding_w2v[i]) * np.array(hx_weights[i]) for i in range(len(embedding_w2v))]
            # embedding_w2v_hx = [a * b for a, b in zip(hx_weights, embedding_w2v)]
            embedding_glove = [W_glove[word_idx_map_glove[word]] for word in text.split(" ")]
            embedding_glove_hx = [np.array(embedding_glove[i]) * np.array(hx_weights[i]) for i in range(len(embedding_glove))]
            # embedding_glove_hx = [a * b for a, b in zip(hx_weights, embedding_glove)]

        sum_hx_weights = sum(hx_weights)
        if sum(hx_weights) == 0:
            doc_embedding.append(doc_embedding[-1])
            continue
        
        embedding_w2v_ = sum(embedding_w2v_hx) / sum_hx_weights
        embedding_glove_ = sum(embedding_glove_hx) / sum_hx_weights
        embedding = [*embedding_w2v_, *embedding_glove_]
        doc_embedding.append(embedding)       
    
    return doc_embedding


def cal_impurity(feature, impurity_criterion="entropy"):
    # y = [rev["y"] for rev in revs]
    y_ser = pd.Series(feature)
    probs = y_ser.value_counts(normalize=True)

    if impurity_criterion == "entropy":
        impurity = -1 * np.sum(np.log2(probs) * probs) 
    elif impurity_criterion == "gini":
        impurity = 1 - np.sum(np.square(probs))

    return impurity


def create_revs_df(revs):
    sentence = []
    y = []
    for rev in revs:
        sentence.append(rev["text"])
        y.append(rev["y"])

    dict = {'sentence': sentence,'y': y}
    revs_df = pd.DataFrame(dict, columns=['sentence', 'y'])
    return revs_df


def information_gain_train(revs):
    vocab = get_train_vocab(revs)
    revs_df = create_revs_df(revs)

    entropy_y = cal_impurity(revs_df["y"])
    ig = dict.fromkeys(vocab, 0)
    for x_value in ig:
        # revs_x_df = revs_df[revs_df['sentence'].str.contains(x_value)]
        x_value_idx = []
        not_x_value_idx = []
        for index, row in revs_df.iterrows():
            text = row['sentence']
            if x_value in text.split(" "):
                x_value_idx.append(index)
            else:
                not_x_value_idx.append(index)

        revs_x_df = revs_df.filter(items=x_value_idx, axis=0)

        conditional_entropy_x = cal_impurity(revs_x_df["y"])
        prob_x = len(revs_x_df) / len(revs_df)
        # revs_not_x_df = revs_df[~revs_df['sentence'].str.contains(x_value)]
        revs_not_x_df = revs_df.filter(items=not_x_value_idx, axis=0)
        conditional_entropy_not_x = cal_impurity(revs_not_x_df["y"])
        prob_not_x = 1 - prob_x
        ig[x_value] = entropy_y - prob_x * conditional_entropy_x - prob_not_x * conditional_entropy_not_x
    
    # print({k: v for k, v in sorted(ig.items(), key=lambda item: item[1])})
    return ig


if __name__ == "__main__":
    datasets = ["rt"] # [ "rt", "cr", "mpqa", "subj"]
    algorithms = ["ig", "se"]
    for dataset in datasets:
        for algorithm in algorithms:
            print("======= training {} dataset by using w2v_glove {} =======".format(dataset, algorithm), flush=True)
            print(flush=True)
            train_model(dataset, algorithm)

    # datasets_dev = ["trec"]  # "sst2", "sst1", "trec"
    # algorithms = ["se"]
    # for dataset in datasets_dev:
    #     for algorithm in algorithms:
    #         print("======= training {} dataset by using w2v_glove {} =======".format(dataset, algorithm), flush=True)
    #         print(flush=True)
    #         train_model_has_dev_set(dataset, algorithm)
