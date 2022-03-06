import argparse
import numpy as np
# from matplotlib import pyplot as plt
import pandas as pd
import scipy.sparse
import scanpy as sc
import os
import anndata
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from evaluation import evaluation
from utils import yaml_config_hook, save_model
from modules import network, mlp, contrastive_loss


def proprocessing():
    sparse_X = scipy.sparse.load_npz('data/filtered_Counts.npz')
    annoData = pd.read_table('data/annoData.txt')
    y = annoData["cellIden"].to_numpy()
    high_var_gene = 5000
    # normlization and feature selection
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
    scDataset = TensorDataset(torch.tensor(x_ndarray, dtype=torch.float32),
                              torch.tensor(y_ndarray, dtype=torch.float32))

    scTrainLength = int(len(scDataset) * 0.8)
    scValidLength = len(scDataset) - scTrainLength
    scTrain, scValid = random_split(scDataset, [scTrainLength, scValidLength])

    scTrainDataLoader = DataLoader(scTrain, shuffle=True, batch_size=64)
    scValidDataLoader = DataLoader(scValid, shuffle=True, batch_size=64)

    for features, labels in scTrainDataLoader:
        print(len(features[-1]))
        print(len(features))
        print(len(labels))
        break

    return scTrainDataLoader, scValidDataLoader


def inference(loader, model, device):
    model.eval()
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            c = model.forward_cluster(x)
        c = c.detach()
        feature_vector.extend(c.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")
    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    # print(feature_vector.shape, labels_vector.shape)
    return feature_vector, labels_vector


# def train():
#     loss_epoch = 0
#     for step, ((x_i, x_j), _) in enumerate(data_loader):
#         optimizer.zero_grad()
#         x_i = x_i.to('cuda')
#         x_j = x_j.to('cuda')
#         z_i, z_j, c_i, c_j = model(x_i, x_j)
#         loss_instance = criterion_instance(z_i, z_j)
#         loss_cluster = criterion_cluster(c_i, c_j)
#         loss = loss_instance + loss_cluster
#         loss.backward()
#         optimizer.step()
#         if step % 10 == 0:
#             print(
#                 f"Step [{step}/{len(data_loader)}]\t loss_instance: {loss_instance.item()}\t loss_cluster: {loss_cluster.item()}")
#         loss_epoch += loss.item()
#     return loss_epoch
def train():
    loss_epoch = 0
    for step, (data, label) in enumerate(data_loader):
        optimizer.zero_grad()
        x_i = data[0].to('cuda')
        x_j = data[1].to('cuda')
        z_i, z_j, c_i, c_j = model(x_i, x_j)
        print(z_i, z_j, c_i, c_j)
        loss_instance = criterion_instance(z_i, z_j)
        loss_cluster = criterion_cluster(c_i, c_j)
        loss = loss_instance + loss_cluster
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(
                f"Step [{step}/{len(data_loader)}]\t loss_instance: {loss_instance.item()}\t loss_cluster: {loss_cluster.item()}")
        loss_epoch += loss.item()
    return loss_epoch

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, Y = inference(test_loader, model, device)
    # print(X.shape,Y.shape)
    nmi, ari, f, acc = evaluation.evaluate(Y, X)
    # print('NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))
    return nmi, ari, f, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_loader, test_loader = proprocessing()
    class_num = args.classnum

    # initialize model
    mlp = mlp.MLP()
    model = network.Network(mlp, args.feature_dim, args.classnum)
    model = model.to('cuda')
    # optimizer / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.reload:
        model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.start_epoch))
        checkpoint = torch.load(model_fp)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1
    loss_device = torch.device("cuda")
    criterion_instance = contrastive_loss.InstanceLoss(args.batch_size, args.instance_temperature, loss_device).to(
        loss_device)
    criterion_cluster = contrastive_loss.ClusterLoss(class_num, args.cluster_temperature, loss_device).to(loss_device)
    # train
    for epoch in range(args.start_epoch, args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train()
        if epoch % 10 == 0:
            save_model(args, model, optimizer, epoch)
        print(f"\nEpoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(data_loader)} \n")
        nmi, ari, f, acc = test()
        print('Test NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))
        print('========' * 8 + '\n')
        # print(f"\nEpoch [{epoch}/{args.epochs}]\t Test Loss: {test_loss_epoch / len(test_loader)} \n")
    save_model(args, model, optimizer, args.epochs)
