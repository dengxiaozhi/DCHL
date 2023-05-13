import os

from torch import nn

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import torch
import torchvision
import argparse
from modules import transform, resnet, network_cross, contrastive_loss
from utils import yaml_config_hook, save_model
from torch.utils import data

from utils.spectral_clustering import KMeans, spectral_clustering


def train():
    loss_epoch = 0
    for step, ((x_i, x_j), _) in enumerate(instance_data_loader):
        optimizer.zero_grad()
        x_i = x_i.to('cuda:0')
        x_j = x_j.to('cuda:0')
        z_i, z_j, c_i, c_j, g_i, g_j = model(x_i, x_j)
        loss_instance = criterion_instance(z_i, z_j)
        loss = loss_instance
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(
                f"Step [{step}/{len(instance_data_loader)}]\t loss_instance: {loss_instance.item()}")
        loss_epoch += loss.item()
    for step, ((x_i, x_j), _) in enumerate(cluster_data_loader):
        optimizer.zero_grad()
        x_i = x_i.to('cuda:0')
        x_j = x_j.to('cuda:0')
        z_i, z_j, c_i, c_j, g_i, g_j = model(x_i, x_j)
        loss_instance = criterion_instance(z_i, z_j)
        loss_cluster = criterion_cluster(c_i, c_j)

        # ===================================================================================
        # K-way normalized cuts or k-Means. Default: k-Means
        use_kmeans = True
        # cluster_num = 200
        cluster_num = max(200, 2 * class_num)
        iter_num = 5
        k_eigen = class_num
        cld_t = 0.2
        if use_kmeans:
            cluster_label1, centroids1 = KMeans(g_i, K=cluster_num, Niters=iter_num)
            cluster_label2, centroids2 = KMeans(g_j, K=cluster_num, Niters=iter_num)
        else:
            cluster_label1, centroids1 = spectral_clustering(g_i, K=k_eigen, clusters=cluster_num, Niters=iter_num)
            cluster_label2, centroids2 = spectral_clustering(g_j, K=k_eigen, clusters=cluster_num, Niters=iter_num)

        # instance-group discriminative learning
        criterion_cld = nn.CrossEntropyLoss().cuda()
        affnity1 = torch.mm(g_j, centroids1.t())
        CLD_loss = criterion_cld(affnity1.div_(cld_t), cluster_label2)

        affnity2 = torch.mm(g_i, centroids2.t())
        CLD_loss = (CLD_loss + criterion_cld(affnity2.div_(cld_t), cluster_label1))/2

        # cross instance-group discriminative learning
        criterion_cross = nn.CrossEntropyLoss().cuda()
        affnity3 = torch.mm(z_j, centroids1.t())  # connect instance head and overcluster head
        cross_loss = criterion_cross(affnity3.div_(cld_t), cluster_label2)

        affnity4 = torch.mm(z_i, centroids2.t())  # connect instance head and overcluster head
        cross_loss = (cross_loss + criterion_cross(affnity4.div_(cld_t), cluster_label1))/2
        # ===================================================================================

        loss = loss_instance + loss_cluster + CLD_loss + cross_loss
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(
                f"Step [{step}/{len(cluster_data_loader)}]\tloss_instance: {loss_instance.item()}\t "
                f"loss_cluster: {loss_cluster.item()}\t loss_cld: {CLD_loss.item()}\t "
                f"loss_cross: {cross_loss.item()}")
        loss_epoch += loss.item()
    return loss_epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config/config_cross.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # prepare data
    if args.dataset == "STL-10":
        train_dataset = torchvision.datasets.STL10(
            root=args.dataset_dir,
            split="train",
            download=True,
            transform=transform.Transforms(size=args.image_size),
        )
        test_dataset = torchvision.datasets.STL10(
            root=args.dataset_dir,
            split="test",
            download=True,
            transform=transform.Transforms(size=args.image_size),
        )
        unlabeled_dataset = torchvision.datasets.STL10(
            root=args.dataset_dir,
            split="unlabeled",
            download=True,
            transform=transform.Transforms(size=args.image_size),
        )
        cluster_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        instance_dataset = unlabeled_dataset
        class_num = 10
    else:
        raise NotImplementedError
    cluster_data_loader = torch.utils.data.DataLoader(
        cluster_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )
    instance_data_loader = torch.utils.data.DataLoader(
        instance_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

    # initialize model
    res = resnet.get_resnet(args.resnet)
    model = network_cross.Network(res, args.feature_dim, class_num)
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
    criterion_cld = nn.CrossEntropyLoss().cuda()
    # train
    for epoch in range(args.start_epoch, args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train()
        if epoch % 20 == 0:
            save_model(args, model, optimizer, epoch)
        print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(instance_data_loader)}")
    save_model(args, model, optimizer, args.epochs)
