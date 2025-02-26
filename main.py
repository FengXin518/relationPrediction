import torch

from models import SpKBGATModified, SpKBGATConvOnly
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from copy import deepcopy

from preprocess import read_entity_from_id, read_relation_from_id, init_embeddings, build_data
from create_batch import Corpus
from utils import save_model
from utils import save_model_last
from utils import load_model
from pathlib import Path

import random
import argparse
import os
import sys
import logging
import time
import pickle

# %%
# %%from torchviz import make_dot, make_dot_from_trace


def parse_args():
    args = argparse.ArgumentParser()
    # network arguments 只能通过长参数获取值
    args.add_argument("-data", "--data",
                      default="./data/WN18RR", help="data directory")
    args.add_argument("-e_g", "--epochs_gat", type=int,
                      default=3600, help="Number of epochs")
    args.add_argument("-e_c", "--epochs_conv", type=int,
                      default=1000, help="Number of epochs")
    args.add_argument("-w_gat", "--weight_decay_gat", type=float,
                      default=5e-6, help="L2 reglarization for gat")
    args.add_argument("-w_conv", "--weight_decay_conv", type=float,
                      default=1e-5, help="L2 reglarization for conv")
    args.add_argument("-pre_emb", "--pretrained_emb", type=bool,
                      default=True, help="Use pretrained embeddings")
    args.add_argument("-emb_size", "--embedding_size", type=int,
                      default=50, help="Size of embeddings (if pretrained not used)")
    args.add_argument("-l", "--lr", type=float, default=1e-3)
    args.add_argument("-g2hop", "--get_2hop", type=bool, default=False)
    args.add_argument("-u2hop", "--use_2hop", type=bool, default=True)
    args.add_argument("-p2hop", "--partial_2hop", type=bool, default=False)
    args.add_argument("-outfolder", "--output_folder",
                      default="./checkpoints/wn/", help="Folder name to save the models.")

    # arguments for GAT
    args.add_argument("-b_gat", "--batch_size_gat", type=int,
                      default=86835, help="Batch size for GAT")
    args.add_argument("-neg_s_gat", "--valid_invalid_ratio_gat", type=int,
                      default=2, help="Ratio of valid to invalid triples for GAT training")
    args.add_argument("-drop_GAT", "--drop_GAT", type=float,
                      default=0.3, help="Dropout probability for SpGAT layer")
    args.add_argument("-alpha", "--alpha", type=float,
                      default=0.2, help="LeakyRelu alphs for SpGAT layer")
    args.add_argument("-out_dim", "--entity_out_dim", type=int, nargs='+',
                      default=[100, 200], help="Entity output embedding dimensions")
    args.add_argument("-h_gat", "--nheads_GAT", type=int, nargs='+',
                      default=[2, 2], help="Multihead attention SpGAT")
    args.add_argument("-margin", "--margin", type=float,
                      default=5, help="Margin used in hinge loss")

    # arguments for convolution network
    args.add_argument("-b_conv", "--batch_size_conv", type=int,
                      default=128, help="Batch size for conv")
    args.add_argument("-alpha_conv", "--alpha_conv", type=float,
                      default=0.2, help="LeakyRelu alphas for conv layer")
    args.add_argument("-neg_s_conv", "--valid_invalid_ratio_conv", type=int, default=40,
                      help="Ratio of valid to invalid triples for convolution training")
    args.add_argument("-o", "--out_channels", type=int, default=500,
                      help="Number of output channels in conv layer")
    args.add_argument("-drop_conv", "--drop_conv", type=float,
                      default=0.0, help="Dropout probability for convolution layer")
    args.add_argument("-ortho_lambda", "--ortho_lambda", type=float,
                      default=0.01, help="Orthogonal regularization coefficient")
    args.add_argument("-patience", "--patience", type=int,default=5, 
                      help="Orthogonal regularization coefficient")
    args.add_argument("-num_heads", "--num_heads", type=int,default=4, 
                      help="最大投影头数量")


    args = args.parse_args()

    #删除之前运行的模型
    subfolders = ["conv/", "Gat/"]
    for subfolder_name in subfolders:
        target_dir = Path(args.output_folder + subfolder_name)
        if not target_dir.exists():
            continue  # 跳过不存在的子文件夹
        
        # 遍历子文件夹内的所有文件并删除
        files_to_delete = [file for file in target_dir.iterdir() if file.is_file()]
        
        if files_to_delete:
            for file in files_to_delete:
                file.unlink()  # 删除文件
    return args

best_loss_conv_model_epoch: int = 0  # 保存最优模型epoch,赋予默认值 0
best_loss_gat_model_epoch: int = 0  # 保存最优模型epoch,赋予默认值 0
args = parse_args()
# %%

def load_data(args):
    train_data, validation_data, test_data, entity2id, relation2id, headTailSelector, unique_entities_train = build_data(
        args.data, is_unweigted=False, directed=True)

    if args.pretrained_emb:
        entity_embeddings, relation_embeddings = init_embeddings(os.path.join(args.data, 'entity2vec.txt'),
                                                                 os.path.join(args.data, 'relation2vec.txt'))
        print("Initialised relations and entities from TransE")

    else:
        entity_embeddings = np.random.randn(
            len(entity2id), args.embedding_size)
        relation_embeddings = np.random.randn(
            len(relation2id), args.embedding_size)
        print("Initialised relations and entities randomly")

    corpus = Corpus(args, train_data, validation_data, test_data, entity2id, relation2id, headTailSelector,
                    args.batch_size_gat, args.valid_invalid_ratio_gat, unique_entities_train, args.get_2hop)

    return corpus, torch.FloatTensor(entity_embeddings), torch.FloatTensor(relation_embeddings)


Corpus_, entity_embeddings, relation_embeddings = load_data(args)


if(args.get_2hop):
    file = args.data + "/2hop.pickle"
    with open(file, 'wb') as handle:
        pickle.dump(Corpus_.node_neighbors_2hop, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)


if(args.use_2hop):
    print("Opening node_neighbors pickle object")
    file = args.data + "/2hop.pickle"
    with open(file, 'rb') as handle:
        node_neighbors_2hop = pickle.load(handle)

entity_embeddings_copied = deepcopy(entity_embeddings)
relation_embeddings_copied = deepcopy(relation_embeddings)

print("Initial entity dimensions {} , relation dimensions {}".format(
    entity_embeddings.size(), relation_embeddings.size()))
# %%

CUDA = torch.cuda.is_available()


def batch_gat_loss(gat_loss_func, train_indices, entity_embed, relation_embed):
    len_pos_triples = int(
        train_indices.shape[0] / (int(args.valid_invalid_ratio_gat) + 1))

    pos_triples = train_indices[:len_pos_triples]
    neg_triples = train_indices[len_pos_triples:]

    pos_triples = pos_triples.repeat(int(args.valid_invalid_ratio_gat), 1)

    source_embeds = entity_embed[pos_triples[:, 0]]
    relation_embeds = relation_embed[pos_triples[:, 1]]
    tail_embeds = entity_embed[pos_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    pos_norm = torch.norm(x, p=1, dim=1)

    source_embeds = entity_embed[neg_triples[:, 0]]
    relation_embeds = relation_embed[neg_triples[:, 1]]
    tail_embeds = entity_embed[neg_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    neg_norm = torch.norm(x, p=1, dim=1)

    y = -torch.ones(int(args.valid_invalid_ratio_gat) * len_pos_triples).cuda()

    loss = gat_loss_func(pos_norm, neg_norm, y)
    return loss


def train_gat(args):

    # Creating the gat model here.
    ####################################
    global best_loss_gat_model_epoch 

    # ------------------------ 模型初始化修改 ------------------------
    print("\nModel type -> GAT with orthogonal projections")
    model_gat = SpKBGATModified(
        entity_embeddings, 
        relation_embeddings,
        args.entity_out_dim,
        args.entity_out_dim,
        args.drop_GAT,
        args.alpha,
        args.nheads_GAT,
        args.num_heads,
        args.ortho_lambda
    )

    if CUDA:
        model_gat.cuda()

    # ------------------------ 优化器配置 ------------------------
    optimizer = torch.optim.Adam(model_gat.parameters(), lr=args.lr, weight_decay=args.weight_decay_gat)

    # ------------------------ 学习率调度器升级 ------------------------
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,
        patience=3,
        verbose=True 
    )

    # ------------------------ 早停机制初始化 ------------------------
    best_val_loss = float('inf')
    early_stop_counter = 0

    gat_loss_func = nn.MarginRankingLoss(margin=args.margin)

    current_batch_2hop_indices = torch.tensor([])
    if(args.use_2hop):
        current_batch_2hop_indices = Corpus_.get_batch_nhop_neighbors_all(args,
                                                                          Corpus_.unique_entities_train, node_neighbors_2hop)

    if CUDA:
        current_batch_2hop_indices = Variable(
            torch.LongTensor(current_batch_2hop_indices)).cuda()
    else:
        current_batch_2hop_indices = Variable(
            torch.LongTensor(current_batch_2hop_indices))

    epoch_losses = []   # losses of all epochs
    print("Number of epochs {}".format(args.epochs_gat))

    for epoch in range(args.epochs_gat):
        print("\nepoch-> ", epoch)
        random.shuffle(Corpus_.train_triples)
        Corpus_.train_indices = np.array(
            list(Corpus_.train_triples)).astype(np.int32)

        model_gat.train()  # getting in training mode
        start_time = time.time()
        epoch_loss = []

        if len(Corpus_.train_indices) % args.batch_size_gat == 0:
            num_iters_per_epoch = len(
                Corpus_.train_indices) // args.batch_size_gat
        else:
            num_iters_per_epoch = (
                len(Corpus_.train_indices) // args.batch_size_gat) + 1

        for iters in range(num_iters_per_epoch):
            start_time_iter = time.time()
            train_indices, train_values = Corpus_.get_iteration_batch(iters)

            if CUDA:
                train_indices = Variable(
                    torch.LongTensor(train_indices)).cuda()
                train_values = Variable(torch.FloatTensor(train_values)).cuda()

            else:
                train_indices = Variable(torch.LongTensor(train_indices))
                train_values = Variable(torch.FloatTensor(train_values))

            # forward pass
            entity_embed, relation_embed,ortho_loss = model_gat(
                Corpus_, Corpus_.train_adj_matrix, train_indices, current_batch_2hop_indices)

            optimizer.zero_grad()

            main_loss = batch_gat_loss(gat_loss_func, train_indices, entity_embed, relation_embed)
            total_loss = main_loss + ortho_loss  # 关键修改点

            total_loss.backward()  # 反向传播总损失
            optimizer.step()

            epoch_loss.append(total_loss.item())  # 记录总损失

            #end_time_iter = time.time()

            # print("Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss {2:.4f}".format(
            #     iters, end_time_iter - start_time_iter, total_loss.data.item()))

        # 学习率调整
        avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
        scheduler.step(avg_epoch_loss)  # 根据平均损失调整学习率

        print("Epoch {} , average loss {} , epoch_time {}".format(
            epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time))
        epoch_losses.append(sum(epoch_loss) / len(epoch_loss))

        #修改
        # 计算指标
        average_loss = sum(epoch_loss) / len(epoch_loss) if epoch_loss else 0
        epoch_time = time.time() - start_time

        # 构建 epoch_info 字典
        epoch_info = {
            "epoch_num": epoch,
            "average_loss": average_loss,
            "epoch_time": epoch_time
        }
        save_model(model_gat, epoch_info, args.output_folder + "Gat/")
        # if epoch % 50 == 0:
        #     save_model(model_gat, epoch_info, args.output_folder + "Gat/")
        # elif epoch == args.epochs_gat - 1:
        #     save_model_last(model_gat, args.data, epoch, args.output_folder + "Gat/")
        
        #早停机制
        if avg_epoch_loss < best_val_loss:
            best_val_loss = avg_epoch_loss
            best_loss_gat_model_epoch = epoch
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= 300:
                print("Early stopping triggered")
                break


def train_conv(args):

    # Creating convolution model here.
    ####################################
    global best_loss_conv_model_epoch

    # ------------------------ 模型初始化修改 ------------------------
    print("Defining model")
    model_gat = SpKBGATModified(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                args.drop_GAT, args.alpha, args.nheads_GAT,args.num_heads,args.ortho_lambda)


    print("Only Conv model trained")
    model_conv = SpKBGATConvOnly(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                 args.drop_GAT, args.drop_conv, args.alpha, args.alpha_conv,
                                 args.nheads_GAT, args.out_channels)

    if CUDA:
        model_conv.cuda()
        model_gat.cuda()

    load_model(model_gat, args.output_folder + "Gat/", best_loss_gat_model_epoch)
    # model_gat.load_state_dict(torch.load(
    #     '{}/trained_{}.pth'.format(args.output_folder + "Gat/", args.epochs_gat - 1)), strict=False)
    model_conv.final_entity_embeddings = model_gat.final_entity_embeddings
    model_conv.final_relation_embeddings = model_gat.final_relation_embeddings

    Corpus_.batch_size = args.batch_size_conv
    Corpus_.invalid_valid_ratio = int(args.valid_invalid_ratio_conv)

    optimizer = torch.optim.Adam(
        model_conv.parameters(), lr=args.lr, weight_decay=args.weight_decay_conv)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=25, gamma=0.5, last_epoch=-1)

    margin_loss = torch.nn.SoftMarginLoss()

    epoch_losses = []   # losses of all epochs
    print("Number of epochs {}".format(args.epochs_conv))

    # ------------------------ 早停机制初始化 ------------------------
    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(args.epochs_conv):
        print("\nepoch-> ", epoch)
        random.shuffle(Corpus_.train_triples)
        Corpus_.train_indices = np.array(
            list(Corpus_.train_triples)).astype(np.int32)

        model_conv.train()  # getting in training mode
        start_time = time.time()
        epoch_loss = []

        if len(Corpus_.train_indices) % args.batch_size_conv == 0:
            num_iters_per_epoch = len(
                Corpus_.train_indices) // args.batch_size_conv
        else:
            num_iters_per_epoch = (
                len(Corpus_.train_indices) // args.batch_size_conv) + 1

        for iters in range(num_iters_per_epoch):
            start_time_iter = time.time()
            train_indices, train_values = Corpus_.get_iteration_batch(iters)

            if CUDA:
                train_indices = Variable(
                    torch.LongTensor(train_indices)).cuda()
                train_values = Variable(torch.FloatTensor(train_values)).cuda()

            else:
                train_indices = Variable(torch.LongTensor(train_indices))
                train_values = Variable(torch.FloatTensor(train_values))

            preds = model_conv(
                Corpus_, Corpus_.train_adj_matrix, train_indices)

            optimizer.zero_grad()

            loss = margin_loss(preds.view(-1), train_values.view(-1))

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data.item())

            # end_time_iter = time.time()

            # print("Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss {2:.4f}".format(
            #     iters, end_time_iter - start_time_iter, loss.data.item()))

        scheduler.step()
        print("Epoch {} , average loss {} , epoch_time {}".format(
            epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time))
        epoch_losses.append(sum(epoch_loss) / len(epoch_loss))

        #修改
        # 计算指标
        average_loss = sum(epoch_loss) / len(epoch_loss) if epoch_loss else 0
        epoch_time = time.time() - start_time

        # 构建 epoch_info 字典
        epoch_info = {
            "epoch_num": epoch,
            "average_loss": average_loss,
            "epoch_time": epoch_time
        }
        
        #保存模型
        save_model(model_conv, epoch_info, args.output_folder + "conv/")
        #早停机制
        if average_loss < best_val_loss:
            best_val_loss = average_loss
            best_loss_conv_model_epoch = epoch
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= 40:
                print("Early stopping triggered")
                break


def evaluate_conv(args, unique_entities):
    model_conv = SpKBGATConvOnly(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                 args.drop_GAT, args.drop_conv, args.alpha, args.alpha_conv,
                                 args.nheads_GAT, args.out_channels)
    load_model(model_conv, args.output_folder + "conv/", best_loss_conv_model_epoch)
    # model_conv.load_state_dict(torch.load(
    #     '{0}conv/trained_{1}.pth'.format(args.output_folder, args.epochs_conv - 1)), strict=False)

    model_conv.cuda()
    model_conv.eval()
    with torch.no_grad():
        Corpus_.get_validation_pred(args, model_conv, unique_entities) 


train_gat(args)

train_conv(args)
evaluate_conv(args, Corpus_.unique_entities_train)
