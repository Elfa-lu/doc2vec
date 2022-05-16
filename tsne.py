import pandas as pd
from utils import *
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def get_data(dataset="rt", algorithm="w2vMean"):
    if dataset == "sst1" or dataset == "sst2" or dataset == "trec":
        revs, W,  _, word_idx_map, vocab, W_glove, word_idx_map_glove, W_ft_wiki, word_idx_map_ft_wiki, W_ft_crawl, word_idx_map_ft_crawl = load_preprocessed_data_sst(dataset)
    else:
        revs, W,  _, word_idx_map, vocab, W_glove, word_idx_map_glove, W_ft_wiki, word_idx_map_ft_wiki, W_ft_crawl, word_idx_map_ft_crawl = load_preprocessed_data(dataset)
    target = [rev["y"] for rev in revs]

    if algorithm == "w2vMean":
        df_feature = get_doc_vec_options(dataset, revs, W, word_idx_map, option="mean")
    elif algorithm == "bow":
        df_feature = get_doc_vec_bow(revs, word_idx_map, vocab)
    elif algorithm == "tfidf":
        df_feature = get_doc_vec_tfidf(revs, word_idx_map, vocab)
    elif algorithm == "glove":
        df_feature = get_doc_vec_options(dataset, revs, W_glove, word_idx_map_glove, option="mean")
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

    x = pd.DataFrame(df_feature)
    y = pd.DataFrame(target, columns=["target"])

    return x, y


def get_tsne(dataset="rt", algorithm="w2vMean", n_components=2, random_state=0):
    x, y = get_data(dataset, algorithm)
    standarized_x = StandardScaler().fit_transform(x)

    if n_components == 2:
        tsne = TSNE(
            n_components=n_components,
            perplexity=100,
            n_iter=3000,
            verbose=1,
            init="pca",
            random_state=random_state,
        )
        z = tsne.fit_transform(standarized_x)
        df = pd.DataFrame()
        df["y"] = y
        df["comp-1"] = z[:, 0]
        df["comp-2"] = z[:, 1]
        n = len(pd.unique(df["y"]))

        sns_plot = sns.scatterplot(
            x="comp-1",
            y="comp-2",
            hue="y",  # df.y.tolist(),
            palette=sns.color_palette("hls", n),
            data=df,
        )  # .set(title="{} data {} T-SNE projection")

        figure = sns_plot.get_figure()
        figure.savefig("{} data {} T-SNE 2d projection.png".format(dataset, algorithm))

        figure.clear()
        plt.close(figure)

    elif n_components == 3:
        tsne = TSNE(
            n_components=n_components,
            perplexity=200,
            n_iter=3000,
            verbose=1,
            init="pca",
            random_state=random_state,
        )
        z = tsne.fit_transform(standarized_x)
        df = pd.DataFrame()
        df["y"] = y
        df["comp-1"] = z[:, 0]
        df["comp-2"] = z[:, 1]
        df["comp-3"] = z[:, 2]
        n = len(pd.unique(df["y"]))

        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.scatter(df["comp-1"], df["comp-2"], df["comp-3"], cmap="viridis")
        ax.set_xlabel("comp-1")
        ax.set_ylabel("comp-2")
        ax.set_zlabel("comp-3")

        plt.savefig("{} data {} T-SNE 3d projection.png".format(dataset, algorithm))


if __name__ == "__main__":
    datasets = ["rt", "trec"]
    algorithms = [
        "w2vMean",
        "glove",
        "fasttext_crawl",
        "glove_w2v_ft",
        "glove_ft",
        "w2v_ft",
        "w2v_glove",
        "bow",
        "tfidf",
    ]

    for dataset in datasets:
        for algorithm in algorithms:
            print("plotting {} by using {}".format(dataset, algorithm))
            get_tsne(dataset, algorithm)
