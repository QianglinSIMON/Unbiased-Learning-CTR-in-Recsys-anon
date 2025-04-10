import torch
from torch import nn, Tensor
from torch.nn import functional as F


class Embedding(nn.Module):
    def __init__(self, feat_sizes: dict, sparse_feature_columns: list,
                 dim_input_dense: int,
                 dense_dim_emb: int,
                 sparse_dim_embs: dict,
                 sparse_output_dim: int,
                 bias: bool = True) -> None:
        """
        注意：sparse_feature_columns的顺序必须与sparse_inputs的列顺序严格一致
        """
        super().__init__()
        self.feat_sizes = feat_sizes
        self.sparse_feature_columns = sparse_feature_columns
        self.dim_input_dense = dim_input_dense
        self.dense_dim_emb = dense_dim_emb
        self.sparse_dim_embs = sparse_dim_embs
        self.sparse_output_dim = sparse_output_dim

        # 稀疏特征嵌入
        self.embedding_dic = nn.ModuleDict({
            feat: nn.Embedding(feat_sizes[feat], sparse_dim_embs[feat])
            for feat in sparse_feature_columns
        })
        for tensor in self.embedding_dic.values():
            nn.init.xavier_normal_(tensor.weight)

        # 稀疏特征统一映射
        self.sparse_to_output = nn.Linear(sum(sparse_dim_embs.values()), sparse_output_dim, bias=bias)
        nn.init.xavier_normal_(self.sparse_to_output.weight)
        if bias: nn.init.zeros_(self.sparse_to_output.bias)

        # 稠密特征映射
        self.dense_embedding = nn.Linear(dim_input_dense, dense_dim_emb, bias=bias)
        nn.init.xavier_normal_(self.dense_embedding.weight)
        if bias: nn.init.zeros_(self.dense_embedding.bias)

    def forward(self, sparse_inputs: Tensor, dense_inputs: Tensor) -> Tensor:
        sparse_outputs = []
        for idx, feat in enumerate(self.embedding_dic.keys()):
            embedded = self.embedding_dic[feat](sparse_inputs[:, idx].long())
            sparse_outputs.append(embedded)

        sparse_concat = torch.cat(sparse_outputs, dim=1)
        sparse_out = self.sparse_to_output(sparse_concat)
        dense_out = self.dense_embedding(dense_inputs)
        return dense_out, sparse_out


class MLP(nn.Module):
    def __init__(self, dim_in, num_hidden, dim_hidden, dim_out=None, batch_norm=True, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_hidden):  # 修正层数构建
            self.layers.append(nn.Linear(dim_in, dim_hidden))
            if batch_norm:
                self.layers.append(DynamicBatchNorm1d(dim_hidden))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
            dim_in = dim_hidden

        # 输出层处理
        if dim_out:
            self.layers.append(nn.Linear(dim_in, dim_out))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DynamicBatchNorm1d(nn.Module):
    """处理单样本和非常规输入情况"""

    def __init__(self, dim_hidden: int):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(dim_hidden)
        self.layer_norm = nn.LayerNorm(dim_hidden)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2 or x.size(0) == 1:  # 处理非常规维度
            return self.layer_norm(x)
        return self.batch_norm(x)


class FeatureSelection(nn.Module):
    def __init__(self, dim_input1, dim_input2, num_hidden=1, dim_hidden=64, dropout=0.0):
        super().__init__()
        self.gate_1 = MLP(dim_input1, num_hidden, dim_hidden, dim_input1,
                          dropout=dropout, batch_norm=True)
        self.gate_2 = MLP(dim_input2, num_hidden, dim_hidden, dim_input2,
                          dropout=dropout, batch_norm=True)

    def forward(self, dense_emb, sparse_emb):
        gate1 = 2.0 * torch.sigmoid(self.gate_1(dense_emb))  # 修正权重缩放
        gate2 = 2.0 * torch.sigmoid(self.gate_2(sparse_emb))
        return gate1 * dense_emb, gate2 * sparse_emb


class Aggregation(nn.Module):
    def __init__(self, dim_inputs_1, dim_inputs_2, num_heads=1):
        super().__init__()
        # 维度校验
        if dim_inputs_1 % num_heads != 0 or dim_inputs_2 % num_heads != 0:
            raise ValueError(f"Input dims ({dim_inputs_1}, {dim_inputs_2}) must be divisible by {num_heads}")

        self.num_heads = num_heads
        self.dim_head_1 = dim_inputs_1 // num_heads
        self.dim_head_2 = dim_inputs_2 // num_heads

        # 参数初始化
        self.w1 = nn.Parameter(torch.Tensor(self.dim_head_1, num_heads, 1))
        self.w2 = nn.Parameter(torch.Tensor(self.dim_head_2, num_heads, 1))
        self.w12 = nn.Parameter(torch.Tensor(num_heads, self.dim_head_1, self.dim_head_2, 1))
        self.bias = nn.Parameter(torch.ones(1, num_heads, 1))
        nn.init.xavier_normal_(self.w1)
        nn.init.xavier_normal_(self.w2)
        nn.init.xavier_normal_(self.w12)

    def forward(self, x1, x2):
        if x1.size(0) == 0 or x2.size(0) == 0:
            return torch.empty(0, 1, device=x1.device)

        # 张量变形
        x1 = x1.view(-1, self.num_heads, self.dim_head_1)
        x2 = x2.view(-1, self.num_heads, self.dim_head_2)

        # 计算项
        term1 = torch.einsum('bhi,iho->bho', x1, self.w1)
        term2 = torch.einsum('bhj,jho->bho', x2, self.w2)
        term12 = torch.einsum('bhi,hijo,bhj->bho', x1, self.w12, x2)

        return torch.sum(term1 + term2 + term12 + self.bias, dim=1)


class FinalMLP(nn.Module):
    def __init__(self, feat_sizes, sparse_feature_columns, dim_input_dense,
                 dense_dim_embedding, sparse_dim_embeddings, sparse_output_dim,
                 dim_hidden_fs=64, num_hidden_1=2, dim_hidden_1=64,
                 num_hidden_2=2, dim_hidden_2=64, num_heads=1, dropout=0.0,
                 l2_reg_fs=0.0, l2_reg_embedding=0.0, l2_reg_mlp=0.0,
                 l2_reg_agg=0.0, device="cpu"):
        super().__init__()
        self.device = device
        self.l2_reg_fs = l2_reg_fs
        self.l2_reg_embedding = l2_reg_embedding
        self.l2_reg_mlp = l2_reg_mlp
        self.l2_reg_agg = l2_reg_agg

        # 嵌入层
        self.embedding = Embedding(feat_sizes, sparse_feature_columns, dim_input_dense,
                                   dense_dim_embedding, sparse_dim_embeddings, sparse_output_dim)

        # 特征选择
        self.feature_selection = FeatureSelection(
            dense_dim_embedding, sparse_output_dim,
            dim_hidden=dim_hidden_fs, dropout=dropout
        )

        # 双塔结构
        self.interaction_1 = MLP(dense_dim_embedding, num_hidden_1, dim_hidden_1)
        self.interaction_2 = MLP(sparse_output_dim, num_hidden_2, dim_hidden_2)

        # 聚合层
        self.aggregation = Aggregation(dim_hidden_1, dim_hidden_2, num_heads)

        # 设备一致性
        self.to(device)

    def forward(self, sparse, dense):
        dense_emb, sparse_emb = self.embedding(sparse, dense)
        fs1, fs2 = self.feature_selection(dense_emb, sparse_emb)
        out1 = self.interaction_1(fs1)
        out2 = self.interaction_2(fs2)
        return self.aggregation(out1, out2)

    def get_regularization_loss(self):
        reg = torch.tensor(0.0, device=self.device)

        # 嵌入层正则
        if self.l2_reg_embedding > 0:
            for emb in self.embedding.embedding_dic.values():
                reg += torch.sum(emb.weight ** 2) * self.l2_reg_embedding
            reg += torch.sum(self.embedding.sparse_to_output.weight ** 2) * self.l2_reg_embedding
            reg += torch.sum(self.embedding.dense_embedding.weight ** 2) * self.l2_reg_embedding

        # 特征选择正则
        if self.l2_reg_fs > 0:
            for p in self.feature_selection.parameters():
                reg += torch.sum(p ** 2) * self.l2_reg_fs

        # MLP正则
        if self.l2_reg_mlp > 0:
            for p in self.interaction_1.parameters():
                reg += torch.sum(p ** 2) * self.l2_reg_mlp
            for p in self.interaction_2.parameters():
                reg += torch.sum(p ** 2) * self.l2_reg_mlp

        # 聚合层正则
        if self.l2_reg_agg > 0:
            for p in self.aggregation.parameters():
                reg += torch.sum(p ** 2) * self.l2_reg_agg

        return reg