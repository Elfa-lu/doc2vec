from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.utils.fixes import loguniform
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
from utils import *
import h5py
import time

# BayesianOptimization
# https://github.com/fmfn/BayesianOptimization/blob/master/examples/sklearn_example.py


def get_data(dataset="rt", algorithm="w2vMean"):
    if dataset == "sst1" or dataset == "sst2" or dataset == "trec":
        revs, W,  _, word_idx_map, vocab, W_glove, word_idx_map_glove, W_ft_wiki, word_idx_map_ft_wiki, W_ft_crawl, word_idx_map_ft_crawl = load_preprocessed_data_sst(dataset)
    else:
        revs, W,  _, word_idx_map, vocab, W_glove, word_idx_map_glove, W_ft_wiki, word_idx_map_ft_wiki, W_ft_crawl, word_idx_map_ft_crawl = load_preprocessed_data(dataset)
    target = [rev["y"] for rev in revs]

    if algorithm == "w2vMean":
        df_feature = get_doc_vec_options(dataset, revs, W, word_idx_map, option="mean")
    elif algorithm == "w2vMax":
        df_feature = get_doc_vec_options(dataset, revs, W, word_idx_map, option="max")
    elif algorithm == "w2vMin":
        df_feature = get_doc_vec_options(dataset, revs, W, word_idx_map, option="min")
    elif algorithm == "bow":
        df_feature = get_doc_vec_bow(revs, word_idx_map, vocab)
    elif algorithm == "tfidf":
        df_feature = get_doc_vec_tfidf(revs, word_idx_map, vocab)
    elif algorithm == "glove":
        df_feature = get_doc_vec_options(dataset, revs, W_glove, word_idx_map_glove, option="mean")
    elif algorithm == "fasttext_wiki":
        df_feature = get_doc_vec_options(dataset, revs, W_ft_wiki, word_idx_map_ft_wiki, option="mean")
    elif algorithm == "fasttext_crawl":
        df_feature = get_doc_vec_options(dataset, revs, W_ft_crawl, word_idx_map_ft_crawl, option="mean")
    elif algorithm == "w2v_glove":
        df_feature = get_doc_vec_multi(dataset, revs, W, word_idx_map, W_glove, word_idx_map_glove)
    elif algorithm == "glove_w2v_ft":
        df_feature = get_doc_vec_multi(dataset, revs, W, word_idx_map, W_glove, word_idx_map_glove, W_ft_crawl, word_idx_map_ft_crawl)
    elif algorithm == "glove_ft":
        df_feature = get_doc_vec_multi(dataset, revs, W_ft_crawl, word_idx_map_ft_crawl, W_glove, word_idx_map_glove)
    elif algorithm == "w2v_ft":
        df_feature = get_doc_vec_multi(dataset, revs, W, word_idx_map, W_ft_crawl, word_idx_map_ft_crawl)
    # lsa_feature = add_lsa_features(revs)
    # print(lsa_feature)
    elif algorithm in ("bert_12", "bert_24", "elmo", "gpt2", "gpt3-babbage", "xlnet", "albert", "gpt3-ada", "roberta"):
        df_feature = get_doc_vec_transformer_sent(dataset, algorithm)

    df = pd.concat([
        pd.DataFrame(target, columns=["target"]), 
        pd.DataFrame(df_feature),
        # pd.DataFrame(lsa_feature),
        ],
        axis=1,
    )

    return df


def train_model(dataset, algorithm, cv=10, random_state=0):
    df = get_data(dataset, algorithm)
    df = shuffle(df, random_state=random_state)
    df.reset_index(inplace=True, drop=True)
    X = df[df.columns.difference(["target"])]
    y = df["target"]

    # gamma: for non-linear hyperplanes, the higher the gamma it tries to exactly fit the data
    # C: penalty of the error term, set the parameter C of class_label to C * value

    # tuned_parameters = [
    #     {"C": loguniform(1e0, 1e3), "gamma": loguniform(1e-4, 1e-1), "kernel": ["rbf"]},
    #     {"kernel": ["linear"], "C": loguniform(1e0, 1e3)},
    #     {"kernel": ["poly"], "degree": [i for i in range(2, 10)]},
    # ]
    # clf = RandomizedSearchCV(SVC(), tuned_parameters, cv=cv, n_iter=15, verbose=3)

    # tuned_parameters = [
    #     {
    #         "kernel": ["rbf"],
    #         "gamma": [1e-1, 1e-2, 1e-3, 1e-4],
    #         "C": [8, 10, 100, 800, 2000],
    #     },
    #     {"kernel": ["linear"], "C": np.logspace(0, 3, num=3, base=15)},
    #     {"kernel": ["poly"], "degree": [i for i in range(2, 10)]},
    # ]

    # clf = GridSearchCV(SVC(), tuned_parameters, scoring="accuracy", verbose=3, cv=cv)
    # clf.fit(X_train, y_train)
    # print("Best parameters set found on development set:")
    # print(clf.best_params_)

    # X_train_, X_test, y_train_, y_test = train_test_split(
    #     X, y, test_size=0.1, random_state=random_state
    # )

    # X_train, X_val, y_train, y_val = train_test_split(
    #     X_train_, y_train_, test_size=0.1, random_state=random_state
    # )

    rows = X.shape[0]
    results = []

    for i in range(cv):
        best_acc = 0
        X_test, y_test = (
            X[rows // cv * i : rows // cv * (i + 1)],
            y[rows // cv * i : rows // cv * (i + 1)],
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X[~X.index.isin(range(rows // cv * i, rows // cv * (i + 1)))],
            y[~y.index.isin(range(rows // cv * i, rows // cv * (i + 1)))],
            test_size=0.1,
            random_state=random_state,
        )

        kernel = "rbf"
        gammas = [1e-1, 1e-2, 1e-3]
        Cs = [10, 100, 1000]
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
                acc = accuracy_score(y_val, clf.predict(X_val))
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
            acc = accuracy_score(y_val, clf.predict(X_val))
            print("--- %s seconds ---" % (time.time() - start_time))
            print("ker:" + str(kernel) + "  degree:" + str(degree), flush=True)
            print("acc:" + str(acc), flush=True)
            print()

            if acc > best_acc:
                best_acc = acc
                best_clf = clf

        best_clf.fit(
            X[~X.index.isin(range(rows // cv * i, rows // cv * (i + 1)))],
            y[~y.index.isin(range(rows // cv * i, rows // cv * (i + 1)))],
        )
        acc = accuracy_score(y_test, best_clf.predict(X_test))
        print("cv: " + str(i + 1) + ", acc: " + str(acc), flush=True)
        results.append(acc)

    print("accuracy for each fold: " + str(results), flush=True)
    print("10 fold vc validation result for SVM: ", str(np.mean(results)), flush=True)


def train_model_has_dev_set(dataset, algorithm, random_state=0, cv=10):
    if dataset == "sst1" or dataset == "sst2":
        file_name = dataset.upper() + ".hdf5"
        f = h5py.File(file_name, "r")
        train_size = f["train_label"].shape[0]
        dev_size = f["dev_label"].shape[0]
        # test_size = f["test_label"].shape[0]

        df = get_data(dataset, algorithm)
        X = df[df.columns.difference(["target"])]
        y = df["target"]

        X_train, X_dev, y_train, y_dev = (
            X[:train_size],
            X[train_size : train_size + dev_size],
            y[:train_size],
            y[train_size : train_size + dev_size],
        )

        X_test, y_test = (
            X[train_size + dev_size :],
            y[train_size + dev_size :],
        )

    elif dataset == "trec":
        file_name = dataset.upper() + ".hdf5"
        f = h5py.File(file_name, "r")
        train_size = f["train_label"].shape[0]
        dev_size = f["test_label"].shape[0]

        df = get_data(dataset, algorithm)
        X = df[df.columns.difference(["target"])]
        y = df["target"]

        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]
        X_train, X_dev, y_train, y_dev = train_test_split(
            X_train, y_train, test_size=0.1, random_state=random_state
        )

        train_size = X_train.shape[0]
        dev_size = X_dev.shape[0]

    best_acc = 0

    kernel = "rbf"
    gammas = [1e-1, 1e-2, 1e-3]
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
    best_clf.fit(X[: train_size + dev_size], y[: train_size + dev_size])
    pred_val = best_clf.predict(X_test)
    acc = accuracy_score(y_test, pred_val)
    print("accuracy on test dataset: " + str(acc))


if __name__ == "__main__":
    # "w2vMean", "w2vMin", "w2vMax", "bow", "tfidf", "w2v_glove", "glove", "se", "ig", 
    # "fasttext_wiki", "fasttext_crawl", "glove_w2v_ft", "glove_ft", "w2v_ft"
    # "bert_12", "bert_24", "elmo", "gpt2", "gpt3-babbage", "xlnet", "albert", "gpt3-ada", "roberta"

    # datasets = ["rt"] # ["rt", "cr", "mpqa", "subj"]
    # algorithms = ["roberta"]  
    # for dataset in datasets:
    #     for algorithm in algorithms:
    #         print("======= training {} dataset by using {} =======".format(dataset, algorithm), flush=True)
    #         print(flush=True)
    #         train_model(dataset, algorithm)

    datasets_dev = ["sst2", "sst1"]  # "sst2", "sst1", "trec" 
    algorithms_dev = ["gpt3-ada"]
    for dataset in datasets_dev:
        for algorithm in algorithms_dev:
            print("======= training {} dataset by using {} =======".format(dataset, algorithm), flush=True)
            print(flush=True)
            train_model_has_dev_set(dataset, algorithm)
