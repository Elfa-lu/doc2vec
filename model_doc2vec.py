from utils import *
import h5py
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import random
import time
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def train_model(dataset, algorithm="doc2vec", cv=10, random_state=0):
    revs, _,  _, _, _, _, _, _, _, _, _ = load_preprocessed_data(dataset)
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

        X_train_ = get_doc2vec(train_)
        X_train = get_doc2vec(train)
        X_dev = get_doc2vec(dev, False)
        X_test = get_doc2vec(test, False)
        
        model = Doc2Vec(X_train, vector_size=300, window=2, epochs=20)
        # model.build_vocab(X_train)
        # model.train(X_train)

        X_train = [model.dv[i] for i in range(len(X_train))]
        X_dev = [model.infer_vector(X_dev[i]) for i in range(len(X_dev))]

        y_train_ = [rev["y"] for rev in train_]
        y_train = [rev["y"] for rev in train]
        y_dev = [rev["y"] for rev in dev]
        y_test = [rev["y"] for rev in test]  

        X_train = pd.DataFrame(X_train)
        X_dev = pd.DataFrame(X_dev)
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

        model_2 = Doc2Vec(X_train_, vector_size=300, window=2, epochs=20)
        X_train_ = [model_2.dv[i] for i in range(len(X_train_))]
        X_test = [model_2.infer_vector(X_test[i]) for i in range(len(X_test))]
        X_train_ = pd.DataFrame(X_train_)
        X_test = pd.DataFrame(X_test)

        best_clf.fit(X_train_, y_train_)
        acc = accuracy_score(y_test, best_clf.predict(X_test))
        print("cv: " + str(i + 1) + ", acc: " + str(acc), flush=True)
        results.append(acc)

    print("accuracy for each fold: " + str(results), flush=True)
    print("10 fold vc validation result for SVM: ", str(np.mean(results)), flush=True)


def train_model_has_dev_set(dataset, algorithm="doc2vec", random_state=0, cv=10):
    revs, _,  _, _, _, _, _, _, _, _, _ = load_preprocessed_data_sst(dataset)

    if dataset == "sst1" or dataset == "sst2":
        file_name = dataset.upper() + ".hdf5"
        f = h5py.File(file_name, "r")
        train_size = f["train_label"].shape[0]
        dev_size = f["dev_label"].shape[0]

        X_train_ = get_doc2vec(revs[:train_size + dev_size])
        X_train = get_doc2vec(revs[:train_size])
        X_dev = get_doc2vec(revs[train_size : train_size + dev_size], False)
        X_test = get_doc2vec(revs[train_size + dev_size :], False)
        
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

        X_train_ = get_doc2vec(revs_train + revs_dev)
        X_train = get_doc2vec(revs_train)
        X_dev = get_doc2vec(revs_dev, False)
        X_test = get_doc2vec(revs[train_size:], False)

        target = [rev["y"] for rev in revs]
        y_train = [element for i, element in enumerate(target) if i in idx_train]
        y_dev = [element for i, element in enumerate(target) if i in idx_dev]
        y_train_ = y_train + y_dev
        y_test = target[train_size:]

    model = Doc2Vec(X_train, vector_size=300, window=5, epochs=20)
    # model.build_vocab(X_train)
    # model.train(X_train, total_examples=model.corpus_count, epochs=model.epochs)

    X_train = [model.dv[i] for i in range(len(X_train))]
    X_dev = [model.infer_vector(X_dev[i]) for i in range(len(X_dev))]

    # train_size = len(y_train)
    # dev_size = len(y_dev)

    X_train = pd.DataFrame(X_train)
    X_dev = pd.DataFrame(X_dev)
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
    model_2 = Doc2Vec(X_train_, vector_size=300, window=5, epochs=20)
    # model.build_vocab(X_train)
    # model.train(X_train, total_examples=model.corpus_count, epochs=model.epochs)

    X_train_ = [model_2.dv[i] for i in range(len(X_train_))]
    X_test = [model_2.infer_vector(X_test[i]) for i in range(len(X_test))]
    X_train_ = pd.DataFrame(X_train_)
    X_test = pd.DataFrame(X_test)

    best_clf.fit(X_train_, y_train_)
    acc = accuracy_score(y_test, best_clf.predict(X_test))
    print("accuracy on test dataset: " + str(acc))


def get_doc2vec(revs, train=True):
    tagged_data = []
    for idx, rev in enumerate(revs):
        text = rev["text"]
        if train:
            tagged_data.append(TaggedDocument(text, [idx]))
        else:
            tagged_data.append(text.split())
    return tagged_data


if __name__ == "__main__":
    datasets = ["mpqa"] # [ "rt", "cr", "mpqa", "subj"]
    algorithms = ["doc2vec"]
    for dataset in datasets:
        for algorithm in algorithms:
            print("======= training {} dataset by using {} =======".format(dataset, algorithm), flush=True)
            print(flush=True)
            train_model(dataset, algorithm)

    # datasets_dev = ["trec"]  # "sst2", "sst1", "trec"
    # algorithms = ["doc2vec"]
    # for dataset in datasets_dev:
    #     for algorithm in algorithms:
    #         print("======= training {} dataset by using {} =======".format(dataset, algorithm), flush=True)
    #         print(flush=True)
    #         train_model_has_dev_set(dataset, algorithm)
