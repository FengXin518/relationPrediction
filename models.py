import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
from layers import SpGraphAttentionLayer, ConvKB

CUDA = torch.cuda.is_available()  # checking cuda availability

"""稀疏图注意力网络（Sparse Graph Attention Network）"""
class SpGAT(nn.Module):
    def __init__(self, num_nodes, nfeat, nhid, relation_dim, dropout, alpha, nheads, num_heads, ortho_lambda):
        """
        参数：
            num_nodes: 图中节点数量
            nfeat: 实体输入特征维度
            nhid: 每头输出的实体隐藏层维度
            relation_dim: 关系嵌入维度
            dropout: Dropout概率
            alpha: LeakyReLU的负斜率
            nheads: 多头注意力头数
            num_heads: 正交投影头数（用于SpGraphAttentionLayer）
            ortho_lambda: 正交正则化系数
        """
        super(SpGAT, self).__init__()
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.nheads = nheads

        # 创建多头注意力层（每个头返回正交损失）
        self.attentions = [
            SpGraphAttentionLayer(
                num_nodes=num_nodes,
                in_features=nfeat,
                out_features=nhid,
                nrela_dim=relation_dim,
                dropout=dropout,
                alpha=alpha,
                concat=True,  # 多头输出拼接
                num_heads=num_heads,
                ortho_lambda=ortho_lambda
            ) for _ in range(nheads)
        ]
        # 将各注意力头注册为子模块
        for i, attention in enumerate(self.attentions):
            self.add_module(f'attention_head_{i}', attention)

        # 输出层注意力（同样返回正交损失）
        self.out_att = SpGraphAttentionLayer(
            num_nodes=num_nodes,
            in_features=nhid * nheads,  # 输入维度为多头拼接后的维度
            out_features=nhid * nheads,  # 输出维度保持相同（后续可接其他层）
            nrela_dim=nhid * nheads,     # 关系维度与实体输出对齐
            dropout=dropout,
            alpha=alpha,
            concat=False,  # 输出层不拼接
            num_heads=num_heads,
            ortho_lambda=ortho_lambda
        )

        # 关系嵌入的线性变换参数
        self.W = nn.Parameter(torch.zeros(size=(relation_dim, nheads * nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

    def forward(self, Corpus_, batch_inputs, entity_embeddings, relation_embed,
                edge_list, edge_type, edge_embed, edge_list_nhop, edge_type_nhop):
        """
        前向传播完整流程：
        1. 处理多跳边的关系嵌入
        2. 多头注意力聚合
        3. 输出层注意力
        4. 关系嵌入变换
        """
        # --------------------- 初始化正交损失 ---------------------
        total_ortho_loss = 0.0

        # --------------------- 处理多跳边关系嵌入 -----------------
        # edge_type_nhop 形状: [num_nhop_edges, 2]
        edge_embed_nhop = relation_embed[edge_type_nhop[:, 0]] + relation_embed[edge_type_nhop[:, 1]]

        # --------------------- 多头注意力处理 ---------------------
        outputs = []
        for att in self.attentions:
            # 每个注意力头返回 (output, ortho_loss)
            h, att_ortho_loss = att(
                input=entity_embeddings,
                edge=edge_list,
                edge_embed=edge_embed,
                edge_list_nhop=edge_list_nhop,
                edge_embed_nhop=edge_embed_nhop
            )
            outputs.append(h)
            total_ortho_loss += att_ortho_loss  # 累加正交损失

        # 拼接多头输出 [num_nodes, nheads * nhid]
        x = torch.cat(outputs, dim=1)
        x = self.dropout_layer(x)  # 应用Dropout

        # --------------------- 输出层注意力 -----------------------
        # 输入: x [num_nodes, nheads*nhid]
        # 注意: edge_embed 使用变换后的关系嵌入
        out_relation_1 = relation_embed.mm(self.W)  # [num_relations, nheads*nhid]
        edge_embed_transformed = out_relation_1[edge_type]  # [num_edges, nheads*nhid]
        edge_embed_nhop_transformed = out_relation_1[edge_type_nhop[:, 0]] + out_relation_1[edge_type_nhop[:, 1]]

        # 输出层注意力（返回最终实体嵌入和正交损失）
        x, out_att_ortho_loss = self.out_att(
            input=x,
            edge=edge_list,
            edge_embed=edge_embed_transformed,
            edge_list_nhop=edge_list_nhop,
            edge_embed_nhop=edge_embed_nhop_transformed
        )
        total_ortho_loss += out_att_ortho_loss  # 累加输出层正交损失

        # --------------------- 返回结果 ---------------------------
        # x: 更新后的实体嵌入 [num_nodes, nhid*nheads]
        # out_relation_1: 变换后的关系嵌入 [num_relations, nhid*nheads]
        # total_ortho_loss: 总正交正则化损失（标量）
        return x, out_relation_1, total_ortho_loss

    def __repr__(self):
        return f"SpGAT(nheads={self.nheads}, in_dim={self.attentions[0].in_features}, out_dim={self.attentions[0].out_features})"


class SpKBGATModified(nn.Module):
    def __init__(self, initial_entity_emb, initial_relation_emb, entity_out_dim, relation_out_dim,
                 drop_GAT, alpha, nheads_GAT,num_heads, ortho_lambda):
        """稀疏KGBAT知识图谱注意力网络初始化
            Args:
                initial_entity_emb (Tensor): 初始实体嵌入 [num_entities, entity_in_dim]
                initial_relation_emb (Tensor): 初始关系嵌入 [num_relations, relation_dim]
                entity_out_dim (list): 各层GAT输出的实体维度，如 [dim1, dim2]
                relation_out_dim (list): 各层GAT输出的关系维度，如 [dim1, dim2]
                drop_GAT (float): GAT层的Dropout概率
                alpha (float): LeakyReLU的负斜率
                nheads_GAT (list): 各层GAT的注意力头数，如 [head1, head2]
        """

        super().__init__()

        self.num_nodes = initial_entity_emb.shape[0]# 实体总数（图中节点数）
        self.entity_in_dim = initial_entity_emb.shape[1]# 实体输入维度（原始嵌入维度）

        # ------------------------- 实体参数初始化 -------------------------
        # 定义两层GAT的输出维度（当前代码仅实现第一层）
        self.entity_out_dim_1 = entity_out_dim[0]# 第一层GAT的每头输出维度
        self.nheads_GAT_1 = nheads_GAT[0]# 第一层GAT的注意力头数
        self.entity_out_dim_2 = entity_out_dim[1]# （预留）第二层GAT的每头输出维度
        self.nheads_GAT_2 = nheads_GAT[1]# （预留）第二层GAT的注意力头数

        # ------------------------- 关系参数初始化 -------------------------
        # Properties of Relations
        self.num_relation = initial_relation_emb.shape[0]# 关系类型总数
        self.relation_dim = initial_relation_emb.shape[1]# 关系输入维度（原始嵌入维度）
        self.relation_out_dim_1 = relation_out_dim[0]# 第一层GAT的关系输出维度

        # ------------------------- 网络超参数 -------------------------
        self.drop_GAT = drop_GAT# GAT层的Dropout概率（通常0.3-0.6）
        self.alpha = alpha      # LeakyReLU的负斜率（通常0.1-0.3）
        self.num_heads = num_heads
        self.ortho_lambda =ortho_lambda

        # ------------------------- 可训练参数定义 -------------------------
        # 最终实体嵌入：多头注意力输出拼接后的维度 = 头数 * 每头维度
        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim_1 * self.nheads_GAT_1)
        )# 形状: [num_entities, entity_out_dim_1 * nheads_GAT_1]
        
        # 最终关系嵌入：与实体嵌入维度对齐，便于后续交互计算
        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1)
        )# 形状: [num_relations, entity_out_dim_1 * nheads_GAT_1]

        # 初始化实体和关系嵌入（可训练参数）
        self.entity_embeddings = nn.Parameter(initial_entity_emb)# 初始实体嵌入
        self.relation_embeddings = nn.Parameter(initial_relation_emb)# 初始关系嵌入

        # ------------------------- 网络层定义 -------------------------
        # 第一层稀疏图注意力网络（当前仅实现单层）
        self.sparse_gat_1 = SpGAT(
            self.num_nodes, # 实体总数
            self.entity_in_dim, # 实体输入维度
            self.entity_out_dim_1, # 每头输出维度
            self.relation_dim,# 关系输入维度
            self.drop_GAT, # Dropout概率
            self.alpha, # LeakyReLU参数
            self.nheads_GAT_1,# 注意力头数
            self.num_heads, 
            self.ortho_lambda
        )

        # ------------------------- 实体嵌入变换矩阵 -------------------------
        # 用于将原始实体嵌入投影到与GAT输出相同维度（残差连接）
        self.W_entities = nn.Parameter(torch.zeros(
            size=(self.entity_in_dim, self.entity_out_dim_1 * self.nheads_GAT_1))
        )# 形状: [entity_in_dim, entity_out_dim_1 * nheads_GAT_1]

        # Xavier均匀初始化（保持输入输出方差一致）
        nn.init.xavier_uniform_(self.W_entities.data, gain=1.414)

    def forward(self, Corpus_, adj, batch_inputs, train_indices_nhop):
        # getting edge list
        # 从邻接矩阵中提取一跳边的连接信息和关系类型
        edge_list = adj[0] # 形状 [2, num_edges], 包含(source, target)节点索引
        edge_type = adj[1] # 形状 [num_edges], 表示每条边的关系类型索引

        # 构建多跳边的连接信息（逆向路径：从尾实体到头实体）
        # train_indices_nhop结构：[batch_size, 4], 每行是 (h, r1, r2, t)
        edge_list_nhop = torch.cat(
            (train_indices_nhop[:, 3].unsqueeze(-1), # 提取尾实体t的列并增加维度
            train_indices_nhop[:, 0].unsqueeze(-1)), # 提取头实体h的列并增加维度
            dim=1).t()
        edge_type_nhop = torch.cat(
            [train_indices_nhop[:, 1].unsqueeze(-1), train_indices_nhop[:, 2].unsqueeze(-1)], dim=1)
        
        # 将数据迁移到GPU（如果可用）
        if(CUDA):
            edge_list = edge_list.cuda()
            edge_type = edge_type.cuda()
            edge_list_nhop = edge_list_nhop.cuda()
            edge_type_nhop = edge_type_nhop.cuda()

        # 获取一跳边的初始关系嵌入
        edge_embed = self.relation_embeddings[edge_type] # 索引操作，形状 [num_edges, relation_dim]

        start = time.time()
        # 对实体嵌入进行L2归一化（梯度截断，仅影响前向传播）
        self.entity_embeddings.data = F.normalize(
            self.entity_embeddings.data, # 输入数据
            p=2,# L2范数
            dim=1# 沿特征维度归一化
            ).detach()# 阻断梯度回传

        # self.relation_embeddings.data = F.normalize(
        #     self.relation_embeddings.data, p=2, dim=1)

        # 通过稀疏图注意力网络更新嵌入
        out_entity_1, out_relation_1, ortho_loss = self.sparse_gat_1(
            Corpus_,# 数据集对象
            batch_inputs,# 当前训练批次的三元组
            self.entity_embeddings, # 归一化后的实体嵌入
            self.relation_embeddings,# 初始关系嵌入
            edge_list, # 一跳边连接
            edge_type, # 一跳边类型
            edge_embed, # 一跳边嵌入
            edge_list_nhop, # 多跳边连接
            edge_type_nhop  # 多跳边类型组合
        )

        # 创建更新掩码：仅更新当前批次涉及的尾实体
        mask_indices = torch.unique(batch_inputs[:, 2]).cuda()      # 获取当前批次的所有尾实体索引
        mask = torch.zeros(self.entity_embeddings.shape[0]).cuda()  # 初始化全零掩码
        mask[mask_indices] = 1.0                                    # 标记需要更新的实体

        # 残差连接：组合原始嵌入和注意力更新结果
        entities_upgraded = self.entity_embeddings.mm(self.W_entities)# 原始嵌入的线性变换
        # 残差连接公式
        out_entity_1 = entities_upgraded + \
            mask.unsqueeze(-1).expand_as(out_entity_1) * out_entity_1# 掩码控制更新范围

        # 对最终实体嵌入进行L2归一化
        out_entity_1 = F.normalize(out_entity_1, p=2, dim=1)

        # 更新模型参数
        self.final_entity_embeddings.data = out_entity_1.data# 存储最终实体嵌入
        self.final_relation_embeddings.data = out_relation_1.data# 存储最终关系嵌入

        return out_entity_1, out_relation_1, ortho_loss# 返回更新后的嵌入


class SpKBGATConvOnly(nn.Module):
    def __init__(self, initial_entity_emb, initial_relation_emb, entity_out_dim, relation_out_dim,
                 drop_GAT, drop_conv, alpha, alpha_conv, nheads_GAT, conv_out_channels):
        """仅包含卷积解码器的稀疏知识图谱网络
            Args:
                initial_entity_emb (Tensor): 初始实体嵌入 [num_entities, entity_in_dim]
                initial_relation_emb (Tensor): 初始关系嵌入 [num_relations, relation_dim]
                entity_out_dim (list): GAT输出的实体维度（未直接使用，保留参数兼容性）
                relation_out_dim (list): GAT输出的关系维度（未直接使用，保留参数兼容性）
                drop_GAT (float): GAT层的Dropout率（未直接使用，保留参数兼容性）
                drop_conv (float): 卷积层的Dropout率
                alpha (float): GAT中LeakyReLU的负斜率（未直接使用，保留参数兼容性）
                alpha_conv (float): 卷积层中LeakyReLU的负斜率
                nheads_GAT (list): GAT的多头数（用于计算嵌入维度）
                conv_out_channels (int): 卷积层的输出通道数
        """

        super().__init__()
        
        # ------------------------- 实体参数初始化 -------------------------
        self.num_nodes = initial_entity_emb.shape[0]        # 实体总数
        self.entity_in_dim = initial_entity_emb.shape[1]    # 实体输入维度（未直接使用）

        # （保留参数，实际未使用）定义两层GAT的输出维度
        self.entity_out_dim_1 = entity_out_dim[0]   # 第一层GAT的每头输出维度
        self.nheads_GAT_1 = nheads_GAT[0]           # 第一层GAT的注意力头数
        self.entity_out_dim_2 = entity_out_dim[1]   # （预留）第二层GAT的每头输出维度
        self.nheads_GAT_2 = nheads_GAT[1]           # （预留）第二层GAT的注意力头数

        # ------------------------- 关系参数初始化 -------------------------
        self.num_relation = initial_relation_emb.shape[0]   # 关系类型总数
        self.relation_dim = initial_relation_emb.shape[1]   # 关系输入维度（未直接使用）
        self.relation_out_dim_1 = relation_out_dim[0]       # （预留）第一层GAT的关系输出维度

        # ------------------------- 网络超参数 -------------------------
        self.drop_GAT = drop_GAT                    # （保留参数，实际未使用）GAT的Dropout率
        self.drop_conv = drop_conv                  # 卷积层的Dropout率
        self.alpha = alpha                          # （保留参数，实际未使用）GAT的LeakyReLU参数
        self.alpha_conv = alpha_conv                # 卷积层的LeakyReLU负斜率
        self.conv_out_channels = conv_out_channels  # 卷积层输出通道数

        # ------------------------- 可训练参数定义 -------------------------
        # 最终实体嵌入（假设已通过GAT预计算）
        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim_1 * self.nheads_GAT_1)
        )# 形状: [num_entities, entity_out_dim_1 * nheads_GAT_1]

        # 最终关系嵌入（假设已通过GAT预计算）
        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1)
        )# 形状: [num_relations, entity_out_dim_1 * nheads_GAT_1]

        # ------------------------- 网络层定义 -------------------------
        # 卷积KB层（用于三元组得分计算）
        self.convKB = ConvKB(
            self.entity_out_dim_1 * self.nheads_GAT_1, # 输入通道数（实体/关系嵌入维度）
            3, 
            1,
            self.conv_out_channels, # 卷积核大小（覆盖三元组的三个元素）
            self.drop_conv,         # Dropout率
            self.alpha_conv         # LeakyReLU参数
        )

    def forward(self, Corpus_, adj, batch_inputs):
        """前向传播（训练阶段）
        Args:
            Corpus_: 数据集对象（未使用，保留参数兼容性）
            adj: 邻接信息（未使用，保留参数兼容性）
            batch_inputs (Tensor): 批量三元组 [batch_size, 3]，每行是 (h, r, t)
        Returns:
            out_conv (Tensor): 三元组得分 [batch_size, 1]
        """
        # 构建卷积输入：将三元组 (h, r, t) 的嵌入拼接为3D张量
        conv_input = torch.cat((self.final_entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1), self.final_relation_embeddings[
            batch_inputs[:, 1]].unsqueeze(1), self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)), dim=1)
        
        # 通过卷积层计算得分
        out_conv = self.convKB(conv_input)# 输出形状: [batch_size, 1]
        return out_conv

    def batch_test(self, batch_inputs):
        """批量测试方法（与forward逻辑相同，独立定义以保持接口清晰）"""
        # 输入处理与forward一致
        conv_input = torch.cat((self.final_entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1), self.final_relation_embeddings[
            batch_inputs[:, 1]].unsqueeze(1), self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)), dim=1)
        out_conv = self.convKB(conv_input)
        return out_conv
