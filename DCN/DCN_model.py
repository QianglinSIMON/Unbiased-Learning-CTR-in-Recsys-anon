import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np


class DeepNet(nn.Module):
    def __init__(self, input_feature_num, dnn_hidden_units, dropout_rate=0.0, device="cpu", init_std=0.0001):
        super().__init__()
        self.device = device
        self.hidden_units = [input_feature_num] + dnn_hidden_units
        self.linears = nn.ModuleList([
            nn.Linear(self.hidden_units[i], self.hidden_units[i + 1]).to(device)
            for i in range(len(self.hidden_units) - 1)
        ])
        self.relus = nn.ModuleList([nn.ReLU() for _ in range(len(self.hidden_units) - 1)])
        self.dropout = nn.Dropout(dropout_rate)

        # 初始化权重
        for linear in self.linears:
            init.normal_(linear.weight, mean=0, std=init_std)
            init.zeros_(linear.bias)

    def forward(self, x):
        x = x.to(self.device)
        for linear, relu in zip(self.linears, self.relus):
            x = linear(x)
            x = relu(x)
            x = self.dropout(x)
        return x


class CrossNet(nn.Module):
    def __init__(self, in_features, layer_num=2, device='cpu'):
        super().__init__()
        self.layer_num = layer_num
        self.kernels = nn.ParameterList([
            nn.Parameter(init.xavier_normal_(torch.empty(in_features, 1)))
            for _ in range(layer_num)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(in_features))
            for _ in range(layer_num)
        ])
        self.to(device)

    def forward(self, x):
        x_0 = x  # [batch_size, in_features]
        x_l = x_0
        for i in range(self.layer_num):
            # 计算交叉项: (x_l^T * w) -> [batch_size, 1]
            xl_w = torch.matmul(x_l, self.kernels[i])  # [batch_size, 1]

            # 外积计算: x_0 * (x_l^T w) -> [batch_size, in_features]
            cross = x_0 * xl_w  # 广播机制实现逐元素乘法

            # 残差连接
            x_l = cross + self.biases[i] + x_l
        return x_l


class CDNet(nn.Module):
    def __init__(self, embedding_index, embedding_size, dense_feature_num, cross_layer_num,
                 dnn_hidden_units, dropout_rate, l2_reg_embedding=0.0, l2_reg_dnn=0.0,
                 l2_reg_dcn=0.0, device="cpu"):
        super().__init__()
        # 参数校验
        if len(embedding_index) != len(embedding_size):
            raise ValueError("embedding_index 和 embedding_size 长度必须一致")

        # 初始化参数
        self.embedding_index = embedding_index
        self.dense_feature_num = dense_feature_num
        self.device = device

        self.l2_reg_embedding=l2_reg_embedding
        self.l2_reg_dnn=l2_reg_dnn
        self.l2_reg_dcn=l2_reg_dcn

        # 嵌入层配置
        embedding_num = [32 if size <= 64 else int(6 * size ** 0.25) for size in embedding_size]
        self.embedding_layers = nn.ModuleList([
            nn.Embedding(size, dim)
            for size, dim in zip(embedding_size, embedding_num)
        ])

        # 初始化嵌入权重
        for emb in self.embedding_layers:
            init.xavier_uniform_(emb.weight)

        # 输入特征维度（稠密特征在前）
        input_feature_num = dense_feature_num + sum(embedding_num)

        # 网络组件
        self.layernorm = nn.LayerNorm(input_feature_num, elementwise_affine=False)
        self.CrossNet = CrossNet(input_feature_num, cross_layer_num, device)
        self.DeepNet = DeepNet(input_feature_num, dnn_hidden_units, dropout_rate, device)

        # 输出层
        self.output_layer = nn.Linear(input_feature_num + dnn_hidden_units[-1], 1)
        self.to(device)

    def forward(self, sparse_feature, dense_feature):
        # 设备一致性
        sparse_feature = sparse_feature.to(self.device)
        dense_feature = dense_feature.to(self.device)

        # 处理稀疏特征
        embedding_feature = []
        for i, feat_idx in enumerate(self.embedding_index):
            input_tensor = sparse_feature[:, feat_idx].long().unsqueeze(1)  # [batch_size, 1]
            embedded = self.embedding_layers[i](input_tensor)  # [batch_size, 1, embedding_dim]
            embedded_mean = embedded.squeeze(1)  # [batch_size, embedding_dim]
            embedding_feature.append(embedded_mean)

        # 拼接特征（稠密在前）
        embedding_feature = torch.cat(embedding_feature, dim=1)
        input_feature = torch.cat((dense_feature, embedding_feature), dim=1)

        # 特征处理
        input_feature = self.layernorm(input_feature)
        out_cross = self.CrossNet(input_feature)
        out_deep = self.DeepNet(input_feature)

        # 输出层
        final_feature = torch.cat((out_cross, out_deep), dim=1)
        pctr = self.output_layer(final_feature).squeeze(-1)
        return pctr

    def get_regularization_loss(self):
        reg_loss = 0.0
        # 嵌入层正则
        if self.l2_reg_embedding > 0:
            for emb in self.embedding_layers:
                reg_loss += self.l2_reg_embedding * torch.sum(emb.weight ** 2)
        # DeepNet 权重正则
        if self.l2_reg_dnn > 0:
            for name, param in self.DeepNet.named_parameters():
                if 'weight' in name:
                    reg_loss += self.l2_reg_dnn * torch.sum(param ** 2)
        # CrossNet 权重正则
        if self.l2_reg_dcn > 0:
            for kernel in self.CrossNet.kernels:
                reg_loss += self.l2_reg_dcn * torch.sum(kernel ** 2)
        return reg_loss