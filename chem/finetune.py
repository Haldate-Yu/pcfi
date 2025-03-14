import argparse
import numpy as np
import os
import pandas as pd
import shutil
import torch
import torch.multiprocessing
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from tensorboardX import SummaryWriter
from torch_geometric.data import DataLoader
from tqdm import tqdm

from loader import MoleculeDataset
from model import GNN_graphpred
from splitters import scaffold_split
from util import load_args

torch.multiprocessing.set_sharing_strategy('file_system')

criterion = nn.BCEWithLogitsLoss(reduction="none")


def train(args, model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        # Whether y is non-null or not.
        is_valid = y ** 2 > 0
        # Loss matrix
        loss_mat = criterion(pred.double(), (y + 1) / 2)
        # loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))

        optimizer.zero_grad()
        loss = torch.sum(loss_mat) / torch.sum(is_valid)
        loss.backward()

        optimizer.step()


def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    # for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    for step, batch in enumerate(loader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = y_true[:, i] ** 2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" % (1 - float(len(roc_list)) / y_true.shape[1]))

    return sum(roc_list) / len(roc_list)  # y_true.shape[1]


def main():
    # Training settings
    args = load_args()

    args.use_early_stopping = args.dataset in ("muv", "hiv")
    args.scheduler = args.dataset in ("bace")

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    # Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "hiv":
        num_tasks = 1
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    else:
        raise ValueError("Invalid dataset name.")

    # set up dataset
    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)

    print(dataset)

    if args.split == "scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,
                                                                    frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    # elif args.split == "random":
    #     train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
    #     print("random")
    # elif args.split == "random_scaffold":
    #     smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
    #     train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
    #     print("random scaffold")
    else:
        raise ValueError("Invalid split option.")

    print(train_dataset[0])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # set up model
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK=args.JK, drop_ratio=args.dropout_ratio,
                          graph_pooling=args.graph_pooling, gnn_type=args.gnn_type)
    if not args.input_model_file == "":
        print("load pretrained model from:", args.input_model_file)
        model.from_pretrained(args.input_model_file)

    model.to(device)

    # set up optimizer
    # different learning rate for different part of GNN
    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    if args.graph_pooling == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr": args.lr * args.lr_scale})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr": args.lr * args.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.3)
    else:
        scheduler = None

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []

    best_val_acc = 0
    final_test_acc = 0

    result_filename = f"results/{args.dataset}_{args.input_model_file}.result"
    fw = open(result_filename, "a")
    fw.write(f"----- seed {args.seed} -------- \n")

    if not args.filename == "":
        fname = 'runs/finetune_cls_runseed' + str(args.runseed) + '/' + args.filename
        # delete the directory if there exists one
        if os.path.exists(fname):
            shutil.rmtree(fname)
            print("removed the existing file.")
        writer = SummaryWriter(fname)

    best_model = None
    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))

        train(args, model, device, train_loader, optimizer)
        if scheduler is not None:
            scheduler.step()

        print("====Evaluation")
        if args.eval_train:
            train_acc = eval(args, model, device, train_loader)
        else:
            print("omit the training accuracy computation")
            train_acc = 0
        val_acc = eval(args, model, device, val_loader)
        test_acc = eval(args, model, device, test_loader)

        if args.use_early_stopping and val_acc > best_val_acc:
            best_val_acc = val_acc
            final_test_acc = test_acc
            best_model = model
        else:
            final_test_acc = test_acc

        print("train: %f val: %f test: %f" % (train_acc, val_acc, final_test_acc))

        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)

        if not args.filename == "":
            writer.add_scalar('data/train auc', train_acc, epoch)
            writer.add_scalar('data/val auc', val_acc, epoch)
            writer.add_scalar('data/test auc', final_test_acc, epoch)

        print("")

    if not args.filename == "":
        writer.close()

    fw.write(f"val: {val_acc_list[-1]}, test: {test_acc_list[-1]}\n")
    fw.write(f"\n\n")
    fw.close()


if __name__ == "__main__":
    main()
