import torch
import torch.nn as nn
from collections import defaultdict

class Wide_deep(nn.Module):
    def __init__(self, feat_sizes, sparse_feature_columns, dense_feature_columns,
                 dnn_hidden_units=[400, 400, 400], dnn_dropout=0.0, embedding_size=4,
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0,
                 init_std=0.0001, seed=1024, device='cpu'):
        super(Wide_deep, self).__init__()
        torch.manual_seed(seed)
        
        # 特征配置校验（修正断言格式）
        assert (len(dense_feature_columns) + len(sparse_feature_columns) == len(feat_sizes)), \
            "特征数量不匹配"
        
        # 设备管理
        self.device = device
        self.dense_feature_columns = dense_feature_columns
        self.sparse_feature_columns = sparse_feature_columns
        
        # Wide部分（仅使用稠密特征）
        self.wide = nn.Linear(len(dense_feature_columns), 1, bias=False)
        
        # Deep部分嵌入层
        self.embedding_dict = nn.ModuleDict({
            feat: nn.Embedding(feat_sizes[feat], embedding_size)
            for feat in sparse_feature_columns
        })
        for emb in self.embedding_dict.values():
            nn.init.normal_(emb.weight, mean=0, std=init_std)
        
        # 特征索引映射修正（稀疏特征从稠密之后开始）
        self.feature_index = {}
        start = len(dense_feature_columns)  # 关键修正点
        for feat in sparse_feature_columns:
            self.feature_index[feat] = start
            start += 1
        
        # DNN结构
        input_size = len(dense_feature_columns) + embedding_size*len(sparse_feature_columns)
        self.dnn = nn.Sequential()
        for i, (in_dim, out_dim) in enumerate(zip([input_size] + dnn_hidden_units, dnn_hidden_units)):
            self.dnn.add_module(f"linear_{i}", nn.Linear(in_dim, out_dim))
            self.dnn.add_module(f"relu_{i}", nn.ReLU())
            self.dnn.add_module(f"dropout_{i}", nn.Dropout(dnn_dropout))
            nn.init.normal_(self.dnn[-3].weight, mean=0, std=init_std)
        self.dnn_out = nn.Linear(dnn_hidden_units[-1], 1, bias=False)
        
        # 偏置项
        self.bias = nn.Parameter(torch.zeros(1))
        
        # 正则化配置
        self.l2_reg_linear = l2_reg_linear
        self.l2_reg_embedding = l2_reg_embedding
        self.l2_reg_dnn = l2_reg_dnn
        
        # 统一设备
        self.to(device)

    def forward(self, x):
        # 输入维度校验（修正断言格式）
        assert (x.shape[1] == len(self.dense_feature_columns)+len(self.sparse_feature_columns)), \
            f"输入特征应为 {len(self.dense_feature_columns)} 稠密 + {len(self.sparse_feature_columns)} 稀疏"
        
        # Wide部分（仅稠密特征）
        wide_input = x[:, :len(self.dense_feature_columns)].float()
        wide_out = self.wide(wide_input)
        
        # Deep部分
        ## 稀疏特征嵌入
        sparse_emb = [
            self.embedding_dict[feat](x[:, idx].long())
            for feat, idx in self.feature_index.items()  # 使用修正后的索引
        ]
        sparse_emb = torch.cat(sparse_emb, dim=1)
        
        ## 拼接稠密特征
        dense_values = x[:, :len(self.dense_feature_columns)].float()
        dnn_input = torch.cat([dense_values, sparse_emb], dim=1)
        
        ## DNN计算
        dnn_out = self.dnn(dnn_input)
        dnn_out = self.dnn_out(dnn_out)
        
        # 最终输出
        return torch.sigmoid(wide_out + dnn_out + self.bias)

    def get_regularization_loss(self):
        reg = 0.0
        
        # Wide部分正则
        if self.l2_reg_linear > 0:
            reg += self.l2_reg_linear * torch.sum(self.wide.weight ** 2)
        
        # 嵌入层正则
        if self.l2_reg_embedding > 0:
            for emb in self.embedding_dict.values():
                reg += self.l2_reg_embedding * torch.sum(emb.weight ** 2)
        
        # DNN正则（包含输出层）
        if self.l2_reg_dnn > 0:
            for layer in self.dnn:
                if isinstance(layer, nn.Linear):
                    reg += self.l2_reg_dnn * torch.sum(layer.weight ** 2)
            reg += self.l2_reg_dnn * torch.sum(self.dnn_out.weight ** 2)
        
        return reg