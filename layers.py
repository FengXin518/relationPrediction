import numpy as np
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from torch.autograd import Variable
from Orthographic import OrthogonalProjectionHeads


CUDA = torch.cuda.is_available()

#模型功能：输入：三元组（如实体和关系嵌入）。
#输出：一个标量值，表示给定三元组的得分（例如，用于判断三元组的真实性）。
#input_seq_len：输入序列长度
#out_chananels：输出通道数
#alpha_leaky:LeakyReLU 的斜率参数
class ConvKB(nn.Module):
    def __init__(self, input_dim, input_seq_len, in_channels, out_channels, drop_prob, alpha_leaky):
        super().__init__()

        self.conv_layer = nn.Conv2d(
            in_channels, out_channels, (1, input_seq_len))  # kernel size -> 1*input_seq_length(i.e. 2)
        self.dropout = nn.Dropout(drop_prob)
        self.non_linearity = nn.ReLU()
        self.fc_layer = nn.Linear((input_dim) * out_channels, 1)

        nn.init.xavier_uniform_(self.fc_layer.weight, gain=1.414)
        nn.init.xavier_uniform_(self.conv_layer.weight, gain=1.414)

    def forward(self, conv_input):

        batch_size, length, dim = conv_input.size()
        # assuming inputs are of the form ->
        conv_input = conv_input.transpose(1, 2)
        # batch * length(which is 3 here -> entity,relation,entity) * dim
        # To make tensor of size 4, where second dim is for input channels
        conv_input = conv_input.unsqueeze(1)

        out_conv = self.dropout(
            self.non_linearity(self.conv_layer(conv_input)))

        input_fc = out_conv.squeeze(-1).view(batch_size, -1)
        output = self.fc_layer(input_fc)
        return output

#适用于边数 E 远小于节点数平方 N^2 的场景
class SpecialSpmmFunctionFinal(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, edge, edge_w, N, E, out_features):
        #构建稀疏张量
        # 非零元素的位置由edge定义，值由edge_w定义
        # assert indices.requires_grad == False
        a = torch.sparse_coo_tensor( 
            edge, edge_w, torch.Size([N, N, out_features]))
        # 对稀疏张量沿 dim=1 求和
        # 沿维度1（目标节点维度）进行聚合求和
        # 将每个源节点到不同目标节点的边权重相加
        # 结果形状变为[N, out_features]
        b = torch.sparse.sum(a, dim=1)
        # 保存上下文信息，用于反向传播
        ctx.N = b.shape[0]
        ctx.outfeat = b.shape[1]
        ctx.E = E
        ctx.indices = a._indices()[0, :]

        #将稀疏结果转换为稠密张量返回
        return b.to_dense()

    @staticmethod
    def backward(ctx, grad_output):
        grad_values = None
        if ctx.needs_input_grad[1]:
            edge_sources = ctx.indices

            if(CUDA):
                edge_sources = edge_sources.cuda()

            # 从 grad_output 中提取与稀疏边相关的梯度
            grad_values = grad_output[edge_sources]
            # grad_values = grad_values.view(ctx.E, ctx.outfeat)
            # print("Grad Outputs-> ", grad_output)
            # print("Grad values-> ", grad_values)
        return None, grad_values, None, None, None

#封装 SpecialSpmmFunctionFinal 的 apply 方法，使其更方便地在模型中使用
class SpecialSpmmFinal(nn.Module):
    def forward(self, edge, edge_w, N, E, out_features):
        return SpecialSpmmFunctionFinal.apply(edge, edge_w, N, E, out_features)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    #num_nodes：图中节点的数量
    #in_features：输入节点特征的维度
    #out_features：输出节点特征的维度
    #nrela_dim：边嵌入的维度，用于增强图结构信息
    #dropout：dropout 概率，用于防止过拟合
    #alpha：LeakyReLU 的负斜率，引入非线性激活
    #concat：是否将多头注意力的输出进行拼接（True）或平均（False）
    #max_proj_heads: 最大投影头数量
    #ortho_lambda: 正交正则化系数
    def __init__(self, num_nodes, in_features, out_features, nrela_dim, dropout, alpha, concat=True, num_heads=4, ortho_lambda=0.01):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.concat = concat
        self.nrela_dim = nrela_dim

        # 新增组件
        self.orth_proj = OrthogonalProjectionHeads(num_heads, in_features)  #引入多正交投影
        # 每个头的注意力参数
        self.attn_params = nn.ParameterList([
            nn.Parameter(torch.zeros(size=(out_features, 2 * in_features + nrela_dim)))
            for _ in range(num_heads)
        ])
        for param in self.attn_params:
            nn.init.xavier_normal_(param.data, gain=1.414)

        #注意力分数进行缩放,线性变化
        self.attn_params1 = nn.ParameterList([
            nn.Parameter(torch.zeros(size=(1, out_features)))
            for _ in range(num_heads)
        ])
        for param in self.attn_params1:
            nn.init.xavier_normal_(param.data, gain=1.414)

        
        # 新增正则化项
        self.ortho_lambda = ortho_lambda  # 正交约束强度系数

        #拼接后线性变换​参数
        self.combine = nn.Linear(num_heads * out_features, out_features)

        # 可学习的加权求和​参数
        self.head_weights = nn.Parameter(torch.ones(num_heads))  

        #自注意力聚合每个正交变换后的特征
        self.aggregate_attn = nn.Linear(out_features, 1)

        # #参数初始化：权重矩阵 self.a 用于计算注意力分数
        # self.a = nn.Parameter(torch.zeros(
        #      size=(out_features, 2 * in_features + nrela_dim)))
        # nn.init.xavier_normal_(self.a.data, gain=1.414)
        # #参数初始化，注意力分数进行缩放,线性变化
        # self.a_2 = nn.Parameter(torch.zeros(size=(1, out_features)))
        # nn.init.xavier_normal_(self.a_2.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm_final = SpecialSpmmFinal() 
        self.num_heads = num_heads

    def _orthogonal_regularization(self):
        """计算正交正则损失"""
        reg_loss = 0
        for mat in self.orth_proj.proj_matrices:
            product = torch.mm(mat, mat.t())
            identity = torch.eye(product.size(0), device=product.device)
            reg_loss += torch.norm(product - identity, p='fro')
        return self.ortho_lambda * reg_loss

    def forward(self, input, edge, edge_embed, edge_list_nhop, edge_embed_nhop):
        #获取输入张量 input 的节点数量 N
        assert input.dim() == 2, "实体嵌入必须是2D张量"
        N = input.size()[0]

        # 多空间正交投影
        input_proj = self.orth_proj(input)  # [N, H, D_out]
        
        # #输出形状
        # print("Input shape:", input.shape) 
        # print("input_proj shape:", input_proj.shape) 

        # Self-attention on the nodes - Shared attention mechanism
        #将一阶边 edge 和多跳边 edge_list_nhop 拼接在一起
        edge = torch.cat((edge[:, :], edge_list_nhop[:, :]), dim=1)
        edge_embed = torch.cat(
            (edge_embed[:, :], edge_embed_nhop[:, :]), dim=0)
        # 多头注意力计算
        head_outputs = []
        for head_idx in range(self.num_heads):
            # 当前头的投影特征
            head_input = input_proj[:, head_idx, :]  # [N, D_out]
            # print("head_input shape:",head_input.shape) 
            # print("value of edge[0]:",edge[0]) 

            # 构造边特征矩阵
            src_nodes = edge[0]  # 源节点索引
            tgt_nodes = edge[1]  # 目标节点索引
            edge_h = torch.cat((
                head_input[src_nodes],  # 源节点特征 [E, D_out]
                head_input[tgt_nodes],  # 目标节点特征 [E, D_out]
                edge_embed             # 边嵌入特征 [E, nrela_dim]
            ), dim=1).t() # [E, 2*D_out + nrela_dim]

            #输出edge_h的形状
            # print("edge_h shape:", edge_h.shape) 
            # print("attn_params[head_idx]",self.attn_params[head_idx].shape)
            
            # 计算注意力分数
            edge_m = torch.matmul(self.attn_params[head_idx], edge_h)  # [E, 1]
            powers = -self.leakyrelu(self.attn_params1[head_idx].mm(edge_m).squeeze())  # [E]
            edge_e = torch.exp(powers).unsqueeze(1)
            
            # 稀疏聚合
            e_rowsum = self.special_spmm_final(edge, edge_e, N, edge_e.size(0), 1)  # [N, 1]
            e_rowsum = torch.clamp(e_rowsum, min=1e-12)  # 防止除零
            e_rowsum = e_rowsum
            # e_rowsum: N x 1
            #压缩 edge_e 的形状，从 (E, 1) 转换为 (E)
            edge_e = edge_e.squeeze(1)
            
            #应用 Dropout，随机丢弃部分注意力权重，防止过拟合
            edge_e = self.dropout(edge_e)
            
            #将注意力权重 edge_e 与边特征 edge_m 相乘，得到边权重 edge_w
            edge_w = (edge_e * edge_m).t()
            
            # 稀疏矩阵乘法聚合邻居信息
            head_output = self.special_spmm_final(edge, edge_w, N, edge_w.shape[0], self.out_features)
            head_outputs.append(head_output)
        
        # 多空间特征聚合
        #平均加和
        #output = torch.mean(torch.stack(head_outputs), dim=0)  # [N, D_out]，原代码

        #拼接后线性变换​
        # stacked = torch.stack(head_outputs)  # [H, N, D_out]
        # concatenated = stacked.permute(1, 0, 2).reshape(N, -1)  # [N, H*D_out]
        # output = self.combine(concatenated)  # [N, D_out]

        #自注意力聚合​（动态权重）0.42
        # stacked = torch.stack(head_outputs)  # [H, N, D_out]
        # attn_scores = self.aggregate_attn(stacked).squeeze(-1)  # [H, N]
        # attn_weights = F.softmax(attn_scores, dim=0)  # [H, N]
        # output = (stacked * attn_weights.unsqueeze(-1)).sum(dim=0)  # [N, D_out]

        #可学习的加权求和​（增加少量参数）
        stacked = torch.stack(head_outputs)  # [H, N, D_out]
        weights = torch.softmax(self.head_weights, dim=0)  # 归一化权重
        output = torch.einsum('h,hnd->nd', weights, stacked)  # 加权求和

        #最大池化0.30
        #output, _ = torch.max(torch.stack(head_outputs), dim=0) 

        assert not torch.isnan(output).any()
        # h_prime: N x out
        #对聚合后的特征进行归一化
        output  = output.div(e_rowsum)
        # h_prime: N x out

        # 添加正则化项
        orth_reg = self._orthogonal_regularization() 
        
        #如果模型不是最后一层（concat=True），应用 ELU 激活函数，引入非线性
        return (F.elu(output) if self.concat else output), orth_reg 
        
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
