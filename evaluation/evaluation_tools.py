'''
Author: Haoze Wu
Date: 2022-04-12 16:34:30
LastEditTime: 2022-04-13 21:54:50
LastEditors: Haoze Wu
Description: Utils that help to generate the clustering over the embeddings using Leiden and KMeans with visualization together
FilePath: \scCC\evaluation\evaluation_tools.py
'''

from sklearn.manifold import TSNE
import seaborn as sns
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             normalized_mutual_info_score, silhouette_score)
from matplotlib import pyplot as plt
import numpy as np


def run_leiden(data, leiden_n_neighbors=300):
    n_pcs = 0
    adata = sc.AnnData(data)
    sc.pp.neighbors(adata, n_neighbors=leiden_n_neighbors, n_pcs=n_pcs, use_rep='X')
    sc.tl.leiden(adata)
    pred = adata.obs['leiden'].to_list()
    pred = [int(x) for x in pred]
    return pred


def cluster_embedding(embedding, cluster_number, Y, save_pred=False, cosine=False,
                      leiden_n_neighbors=50, cluster_methods=["KMeans", "Leiden"]):
    result = {}
    if "KMeans" in cluster_methods:
        # evaluate K-Means
        kmeans = KMeans(n_clusters=cluster_number,
                        init="k-means++",
                        random_state=0)
        if cosine:
            from sklearn import preprocessing
            embedding_norm = preprocessing.normalize(embedding)
            pred = kmeans.fit_predict(embedding_norm)
        else:
            pred = kmeans.fit_predict(embedding)
        if Y is not None:
            result[f"kmeans_ari"] = adjusted_rand_score(Y, pred)
            result[f"kmeans_nmi"] = normalized_mutual_info_score(Y, pred)
        result[f"kmeans_sil"] = silhouette_score(embedding, pred)
        result[f"kmeans_cal"] = calinski_harabasz_score(embedding, pred)
        if save_pred:
            result[f"kmeans_pred"] = pred
    if "Leiden" in cluster_methods:
        # evaluate leiden
        pred = run_leiden(embedding, leiden_n_neighbors)

        if Y is not None:
            result[f"leiden_ari"] = adjusted_rand_score(Y, pred)
            result[f"leiden_nmi"] = normalized_mutual_info_score(Y, pred)
        result[f"leiden_sil"] = silhouette_score(embedding, pred)
        result[f"leiden_cal"] = calinski_harabasz_score(embedding, pred)
        if save_pred:
            result[f"leiden_pred"] = pred

    return result


def embedding_cluster_visualization(embeddings, result, true_label, args, epoch):
    X_tsne = TSNE(n_components=2, random_state=24).fit_transform(embeddings)
    f, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))
    palette_0 = sns.color_palette("tab20", args.classnum)
    axs = sns.scatterplot(X_tsne[:, 0], X_tsne[:, 1], hue=result["kmeans_pred"] + 1, legend='full', palette=palette_0,
                          ax=ax[0])
    sns.move_legend(ax[0], "upper left", bbox_to_anchor=(1, 1))
    axs.set_title(
        f'Epoch {epoch} with KMeans predict label\n' + f"ARI:{round(result['kmeans_ari'], 2)}  " + f"NMI:{round(float(result['kmeans_nmi']), 2)} ")

    palette_1 = sns.color_palette("tab20", len(np.unique(result["leiden_pred"])))
    axs = sns.scatterplot(X_tsne[:, 0], X_tsne[:, 1], hue=np.array(result["leiden_pred"]) + 1, legend='full',
                          palette=palette_1, ax=ax[1])
    sns.move_legend(ax[1], "upper left", bbox_to_anchor=(1, 1))
    axs.set_title(
        f'Epoch {epoch} with Leiden predict label\n' + f"ARI:{round(result['leiden_ari'], 2)}  " + f"NMI:{round(float(result['leiden_nmi']), 2)} ")

    axs = sns.scatterplot(X_tsne[:, 0], X_tsne[:, 1], hue=true_label, legend='full', palette=palette_0, ax=ax[2])
    sns.move_legend(ax[2], "upper left", bbox_to_anchor=(1, 1))
    axs.set_title(f'Epoch {epoch} with true label')

    plt.show()
