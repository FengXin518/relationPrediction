import torch
# from torchviz import make_dot, make_dot_from_trace
from models import SpKBGATModified, SpKBGATConvOnly
from layers import ConvKB
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from copy import deepcopy

from preprocess import read_entity_from_id, read_relation_from_id, init_embeddings, build_data
from create_batch import Corpus

import random
import argparse
import os
import logging
import time
import pickle
import glob 


CUDA = torch.cuda.is_available()


def save_model_last(model, name, epoch, folder_name):
    print("Saving Model")
    torch.save(model.state_dict(),
               (folder_name + "trained_{}.pth").format(epoch))
    print("Done saving Model")

def save_model(model, epoch_info, folder_name):
    """
    保存模型，文件名包含 epoch、average loss 和 epoch_time
    参数:
        model: 要保存的模型
        name: 数据集名称（用于日志）
        epoch_info: 字典，包含 epoch_num, average_loss, epoch_time
        folder_name: 保存目录
    """
    # 确保目录存在
    os.makedirs(folder_name, exist_ok=True)
    
    # 从 epoch_info 中提取信息
    epoch = epoch_info.get("epoch_num", 0)
    avg_loss = epoch_info.get("average_loss", 0)
    time_spent = epoch_info.get("epoch_time", 0)

    # 生成文件名（示例：epoch5_loss0.1234_time120s.pth）
    filename = f"epoch{epoch}_loss{avg_loss:.4f}_time{time_spent:.0f}s.pth"
    save_path = os.path.join(folder_name, filename)

    # 保存模型
    print(f"Saving Model: {filename}")
    torch.save(model.state_dict(), save_path)
    #print(f"Done saving Model to {save_path}")

# 加载模型
def load_model(model, save_dir, target_epoch):
    # 构造匹配模式（注意没有前导零填充和下划线）
    pattern = f"epoch{target_epoch}_*.pth"
    model_path = os.path.join(save_dir, pattern)
    print(f"[Debug] 模型查找路径: {model_path}") 
    
    # 使用 glob 查找文件
    matched_files = glob.glob(model_path)
    if not matched_files:
        raise FileNotFoundError(f"找不到 epoch={target_epoch} 的模型文件")
    
    # 加载第一个匹配的模型（建议添加排序确保选择最新）
    model.load_state_dict(torch.load(matched_files[0]))
    print(f"已加载模型: {matched_files[0]}")
    return model

gat_loss_func = nn.MarginRankingLoss(margin=0.5)


def GAT_Loss(train_indices, valid_invalid_ratio):
    len_pos_triples = train_indices.shape[0] // (int(valid_invalid_ratio) + 1)

    pos_triples = train_indices[:len_pos_triples]
    neg_triples = train_indices[len_pos_triples:]

    pos_triples = pos_triples.repeat(int(valid_invalid_ratio), 1)

    source_embeds = entity_embed[pos_triples[:, 0]]
    relation_embeds = relation_embed[pos_triples[:, 1]]
    tail_embeds = entity_embed[pos_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    pos_norm = torch.norm(x, p=2, dim=1)

    source_embeds = entity_embed[neg_triples[:, 0]]
    relation_embeds = relation_embed[neg_triples[:, 1]]
    tail_embeds = entity_embed[neg_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    neg_norm = torch.norm(x, p=2, dim=1)

    y = torch.ones(int(args.valid_invalid_ratio)
                   * len_pos_triples).cuda()
    loss = gat_loss_func(pos_norm, neg_norm, y)
    return loss


def render_model_graph(model, Corpus_, train_indices, relation_adj, averaged_entity_vectors):
    graph = make_dot(model(Corpus_.train_adj_matrix, train_indices, relation_adj, averaged_entity_vectors),
                     params=dict(model.named_parameters()))
    graph.render()


def print_grads(model):
    print(model.relation_embed.weight.grad)
    print(model.relation_gat_1.attention_0.a.grad)
    print(model.convKB.fc_layer.weight.grad)
    for name, param in model.named_parameters():
        print(name, param.grad)


def clip_gradients(model, gradient_clip_norm):
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, "norm before clipping is -> ", param.grad.norm())
            torch.nn.utils.clip_grad_norm_(param, args.gradient_clip_norm)
            print(name, "norm beafterfore clipping is -> ", param.grad.norm())


def plot_grad_flow(named_parameters, parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []

    for n, p in zip(named_parameters, parameters):
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="r")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="g")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="r", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="g", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig('initial.png')
    plt.close()


def plot_grad_flow_low(named_parameters, parameters):
    ave_grads = []
    layers = []
    for n, p in zip(named_parameters, parameters):
        # print(n)
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig('initial.png')
    plt.close()
