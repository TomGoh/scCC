import scipy.sparse
import scanpy as sc
import anndata
import pandas as pd
import numpy as np

def data_process(x_path,y_path,args):

    sparse_X = scipy.sparse.load_npz(x_path)
    annoData = pd.read_table(y_path)
    y = annoData["cellIden"].to_numpy()
    high_var_gene = args.num_genes

    adataSC = anndata.AnnData(X=sparse_X, obs=np.arange(sparse_X.shape[0]), var=np.arange(sparse_X.shape[1]))
    sc.pp.filter_genes(adataSC, min_cells=10)
    adataSC.raw = adataSC
    sc.pp.highly_variable_genes(adataSC, n_top_genes=high_var_gene, flavor='seurat_v3')
    sc.pp.normalize_total(adataSC, target_sum=1e4)
    sc.pp.log1p(adataSC)

    adataNorm = adataSC[:, adataSC.var.highly_variable]
    dataframe = adataNorm.to_df()
    x_ndarray = dataframe.values.squeeze()
    y_ndarray = np.expand_dims(y, axis=1)
    print(f'X Shape: {x_ndarray.shape}, Y Shape: {y_ndarray.shape}')
    return x_ndarray,y_ndarray