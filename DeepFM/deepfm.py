import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import sys
import os
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict, namedtuple, defaultdict
import random


class FM(nn.Module):
    def __init__(self, p, k, device):
        super(FM, self).__init__()
        self.p = p  # 输入特征的维度
        self.k = k  # 潜在因子维度
        self.device = device

        # 线性部分
        self.linear = nn.Linear(self.p, 1, bias=True)

        # 隐因子矩阵
        self.v = nn.Parameter(torch.Tensor(self.p, self.k), requires_grad=True)
        nn.init.xavier_normal_(self.v)

        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = x.to(self.device)
        linear_part = self.linear(x)

        # 优化后的二阶交互计算
        xv = torch.mm(x, self.v)
        inter_part = 0.5 * torch.sum(xv ** 2 - torch.mm(x ** 2, self.v ** 2), dim=1, keepdim=True)

        inter_part = self.drop(inter_part)
        output = linear_part + inter_part
        return output.view(-1, 1)


class deepfm(nn.Module):
    def __init__(self, feat_sizes, sparse_feature_columns, dense_feature_columns,
                 dnn_hidden_units=[400, 400, 400], dnn_dropout=0.0, embedding_size=4,
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0,
                 init_std=0.0001, seed=1024, device='cpu'):

        super(deepfm, self).__init__()
        torch.manual_seed(seed)
        self.device = device
        self.sparse_feature_columns = sparse_feature_columns
        self.dense_feature_columns = dense_feature_columns
        self.embedding_size = embedding_size

        # 修正特征索引构建顺序
        self.feature_index = {}
        start = 0
        for feat in  self.dense_feature_columns+self.sparse_feature_columns: #先处理 dense 特征，然后是 sparse 特征
            self.feature_index[feat] = start
            start += 1

        # 初始化嵌入层
        self.embedding_dic = nn.ModuleDict({
            feat: nn.Embedding(feat_sizes[feat], embedding_size)
            for feat in self.sparse_feature_columns
        })
        
        for emb in self.embedding_dic.values():
            nn.init.xavier_normal_(emb.weight)

        # 输入维度计算
        self.input_size = embedding_size * len(sparse_feature_columns) + len(dense_feature_columns)

        # FM部分
        self.fm = FM(self.input_size, 10, device=device)

        # DNN部分
        self.dnn = nn.Sequential()
        hidden_units = [self.input_size] + dnn_hidden_units

        for i in range(len(hidden_units) - 1):
            linear_layer = nn.Linear(hidden_units[i], hidden_units[i + 1])
            nn.init.kaiming_normal_(linear_layer.weight, mode='fan_in', nonlinearity='relu')
    
            # 按顺序添加各层
            self.dnn.add_module(f'linear_{i}', linear_layer)
            self.dnn.add_module(f'relu_{i}', nn.ReLU())
            self.dnn.add_module(f'dropout_{i}', nn.Dropout(dnn_dropout))

        # 初始化输出层
        self.dnn_out = nn.Linear(dnn_hidden_units[-1], 1, bias=False)
        nn.init.xavier_normal_(self.dnn_out.weight)  # 添加输出层初始化

        self.bias = nn.Parameter(torch.zeros((1,)))

        # 正则化参数
        self.l2_reg_linear = l2_reg_linear
        self.l2_reg_embedding = l2_reg_embedding
        self.l2_reg_dnn = l2_reg_dnn

        # 统一设备管理
        self.to(device)

    def forward(self, x):
        x = x.to(self.device)

        # 处理稀疏特征
        sparse_emb = [self.embedding_dic[feat](x[:, self.feature_index[feat]].long())
                      for feat in self.sparse_feature_columns]
        sparse_emb = torch.cat(sparse_emb, dim=-1)

        # 处理稠密特征
        dense_values = [x[:, self.feature_index[feat]].unsqueeze(1)
                        for feat in self.dense_feature_columns]
        dense_input = torch.cat(dense_values, dim=1).float() if dense_values else torch.empty(x.size(0), 0)

        # 合并特征
        combined = torch.cat([dense_input, sparse_emb], dim=1) if dense_values else sparse_emb

        # FM部分
        fm_output = self.fm(combined)

        # DNN部分
        dnn_output = self.dnn(combined)
        
        dnn_output = self.dnn_out(dnn_output)

        # 最终输出
        output = torch.sigmoid(fm_output + dnn_output + self.bias)
        return output

    def get_regularization_loss(self):
        reg_loss = 0.0

        # FM正则
        if self.l2_reg_linear > 0:
            reg_loss += torch.norm(self.fm.linear.weight, p=2) ** 2 * self.l2_reg_linear
        if self.l2_reg_embedding > 0:
            reg_loss += torch.norm(self.fm.v, p=2) ** 2 * self.l2_reg_embedding

        # 嵌入层正则
        if self.l2_reg_embedding > 0:
            for emb in self.embedding_dic.values():
                reg_loss += torch.norm(emb.weight, p=2) ** 2 * self.l2_reg_embedding

        # DNN正则
        if self.l2_reg_dnn > 0:
            for layer in self.dnn:
                if isinstance(layer, nn.Linear):
                    reg_loss += torch.norm(layer.weight, p=2) ** 2 * self.l2_reg_dnn
            reg_loss += torch.norm(self.dnn_out.weight, p=2) ** 2 * self.l2_reg_dnn

        return reg_loss
