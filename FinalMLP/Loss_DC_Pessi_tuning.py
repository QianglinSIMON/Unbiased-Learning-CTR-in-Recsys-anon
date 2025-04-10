import torch
from final_mlp import FinalMLP
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import torch.nn as nn
from scipy.stats import norm
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Data_preprocess_split import (generate_data, get_metrics, create_data_loaders,
                                   custom_loss_function, EarlyStopping,get_sparse_emb_dims)

import time

# 记录开始时间
start_time = time.time()
# 定义超参数
lr = 1e-5
batch_size = 1024
wd = 0
epochs = 500
l2_reg_mlp=1e-4
l2_reg_agg=5e-4
l2_reg_fs=5e-4
l2_reg_embedding = 5e-4
patience = 20
dense_dim_embedding =128
sparse_output_dim =128
dim_hidden_fs=64
num_hidden_1=2
dim_hidden_1=256
num_hidden_2=2
dim_hidden_2=256
num_heads=2
dropout = 0.8
seed = 2028


torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
np.random.seed(seed)
random.seed(seed)

tuning_percentage = 0.2
# 调用 generate_data 函数，获取数据集和微调比例
df_biased, df_uniform, df_test, df_val, df_uniform_train, sparse_features, dense_features, feat_sizes, feature_names = generate_data(
    seed=seed)

# 随机抽取一定百分比的无偏数据用于微调
sample_size = int(len(df_uniform_train) * tuning_percentage)
df_tuning = df_uniform_train.sample(n=sample_size, replace=False, random_state=seed)

target = ['is_click']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sparse_feature_columns = sparse_features
dense_feature_columns = dense_features

# 获取嵌入维度字典
sparse_dim_embeddings = get_sparse_emb_dims(feat_sizes, sparse_features)

# 创建数据加载器
test_loader = create_data_loaders(batch_size=batch_size, dataset=df_test, sparse_features=sparse_features,
                                  dense_features=dense_features, target=target, shuffle=False)
val_loader = create_data_loaders(batch_size=batch_size, dataset=df_val, sparse_features=sparse_features,
                                 dense_features=dense_features, target=target, shuffle=False)

# 设定目标批次数
tuning_batch_size = 256

# 根据目标批次数计算批次大小
batch_size_train = batch_size_surrogate = tuning_batch_size

# 准备数据加载器
train_loader = create_data_loaders(batch_size=batch_size_train, dataset=df_biased, sparse_features=sparse_features,
                                   dense_features=dense_features, target=target, shuffle=True)

surrogate_loader = create_data_loaders(batch_size=batch_size_surrogate, dataset=df_tuning,
                                       sparse_features=sparse_features, dense_features=dense_features, target=target,
                                       shuffle=True)

# 创建模型实例
model = FinalMLP(
    feat_sizes=feat_sizes,
    sparse_feature_columns=sparse_feature_columns,
    dim_input_dense=len(dense_features),
    dense_dim_embedding=dense_dim_embedding,
    sparse_dim_embeddings=sparse_dim_embeddings,
    sparse_output_dim=sparse_output_dim,
    dim_hidden_fs=dim_hidden_fs,
    num_hidden_1=num_hidden_1,
    dim_hidden_1=dim_hidden_1,
    num_hidden_2=num_hidden_2,
    dim_hidden_2=dim_hidden_2,
    num_heads=num_heads,
    dropout=dropout,
    l2_reg_fs=l2_reg_fs,
    l2_reg_embedding=l2_reg_embedding,
    l2_reg_mlp=l2_reg_mlp,
    l2_reg_agg=l2_reg_agg,
    device=device
).to(device)


# 加载预训练的模型参数
model.load_state_dict(torch.load('pre_trained_model_{}.pth'.format(int(seed)), map_location=device, weights_only=False))

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

auc_score_list = []
precision_score_list = []
recall_score_list = []
f1_score_list = []
NDCG_score_list = []
epoch_list = []
train_loss_list = []  # 保存训练集上的平均损失
test_loss_list = []  # 保存测试集上的平均损失

mse_loss_fn = nn.BCELoss(reduction='none')  # 使用 BCEMSE 损失计算
# 初始化早停
early_stopping = EarlyStopping(patience=patience, min_delta=0.001,
                               save_path="Loss_DC_Pessi_fintuning_model_{}_percent.pth".format(
                                   int(tuning_percentage * 100)))

best_alpha_matrix = []  # 用于记录每个 epoch 每个 batch 下的最佳权重

# 中心化 loss 列，如果只有一行数据，直接设置为0
def centralize_loss(df):
    if len(df) == 1:
        df['centralized_loss'] = 0  # 如果只有一行数据，直接将 loss 设为 0
    else:
        mean_loss = df['loss'].mean()
        df['centralized_loss'] = df['loss'] - mean_loss  # 其他情况下按均值中心化
    return df


for epoch in range(epochs):
    model.train()

    total_loss_epoch = 0.0  # 初始化总损失
    total_batches = 0  # 初始化批次数

    # 创建迭代器
    train_loader_iter = iter(train_loader)
    surrogate_loader_iter = iter(surrogate_loader)
    best_alpha_list = []  # 当前 epoch 的 batch 权重列表

    for (train_sparse_inputs, train_dense_inputs, train_labels), (
            surrogate_sparse_inputs, surrogate_dense_inputs, surrogate_labels) in zip(train_loader_iter,
                                                                                      surrogate_loader_iter):
        optimizer.zero_grad()

        train_sparse_inputs = train_sparse_inputs.to(device)
        train_dense_inputs = train_dense_inputs.to(device)
        train_labels = train_labels.to(device).float()

        y_hat_train = model(train_sparse_inputs, train_dense_inputs)

        y_hat_train = torch.sigmoid(y_hat_train)  # 确保输出在0和1之间

        # 使用自定义损失函数
        loss_train = custom_loss_function(y_hat_train, train_labels, model,
                                          l2_reg_mlp=model.l2_reg_mlp, l2_reg_agg=model.l2_reg_agg,
                                          l2_reg_fs=model.l2_reg_fs, l2_reg_embedding=model.l2_reg_embedding)

        surrogate_sparse_inputs = surrogate_sparse_inputs.to(device)
        surrogate_dense_inputs = surrogate_dense_inputs.to(device)
        surrogate_labels = surrogate_labels.to(device).float()
        y_hat_surrogate = model(surrogate_sparse_inputs, surrogate_dense_inputs)
        y_hat_surrogate = torch.sigmoid(y_hat_surrogate)  # 确保输出在0和1之间
        loss_surrogate = custom_loss_function(y_hat_surrogate, surrogate_labels, model,
                                              l2_reg_mlp=model.l2_reg_mlp, l2_reg_agg=model.l2_reg_agg,
                                              l2_reg_fs=model.l2_reg_fs, l2_reg_embedding=model.l2_reg_embedding)

        # 将 x_train 和 y_train 合并为一个数据集
        train_batch_data = torch.cat((train_dense_inputs, train_sparse_inputs, train_labels), dim=1)
        # 使用 feature_names 和 label 列名构造数据集
        columns = feature_names + target
        train_batch_df = pd.DataFrame(train_batch_data.cpu().numpy(), columns=columns)

        # 将 x_surrogate 和 y_surrogate 合并为一个数据集
        surrogate_batch_data = torch.cat((surrogate_dense_inputs, surrogate_sparse_inputs, surrogate_labels), dim=1)

        # 使用 feature_names 和 label 列名构造数据集
        columns = dense_features + sparse_features + target
        surrogate_batch_df = pd.DataFrame(surrogate_batch_data.cpu().numpy(), columns=columns)

        # 为数据集添加来源标签
        train_batch_df['source'] = 'biased'
        surrogate_batch_df['source'] = 'unbiased'

        data_loss_train  = mse_loss_fn(y_hat_train.view(-1), train_labels.view(-1))  # 将形状调整为一维
        train_batch_df['loss'] = data_loss_train.detach().cpu().numpy()  # 将损失值添加到有偏数据集

        data_loss_surrogate = mse_loss_fn(y_hat_surrogate.view(-1), surrogate_labels.view(-1))  # 将形状调整为一维
        surrogate_batch_df['loss'] = data_loss_surrogate.detach().cpu().numpy()  # 将损失值添加到无偏数据集

        # 提取有偏数据集的指定列
        train_selected_df = train_batch_df.loc[:, ['user_id', 'video_id', 'source', 'loss']]

        # 提取无偏数据集的指定列
        surrogate_selected_df = surrogate_batch_df.loc[:, ['user_id', 'video_id', 'source', 'loss']]

        train_selected= centralize_loss(train_selected_df)
        surrogate_selected = centralize_loss(surrogate_selected_df)

        # 合并两个数据集，标记所有可能的匹配关系
        merged = pd.merge(
            train_selected,
            surrogate_selected,
            on=['user_id', 'video_id'],
            how='outer',
            suffixes=('_train', '_surrogate')
        )

        # 分组计算协方差贡献
        # Group 1: user_id 和 video_id 完全匹配
        group1_mask = merged['user_id'].notna() & merged['video_id'].notna()
        group1_cov = (merged.loc[group1_mask, 'centralized_loss_train'] * merged.loc[
            group1_mask, 'centralized_loss_surrogate']).sum()

        # Group 2: 仅 user_id 匹配（video_id 不匹配）
        merged_user = pd.merge(
            train_selected,
            surrogate_selected,
            on='user_id',
            how='inner',
            suffixes=('_train', '_surrogate')
        )
        group2_mask = merged_user['video_id_train'] != merged_user['video_id_surrogate']
        group2_cov = (merged_user.loc[group2_mask, 'centralized_loss_train'] * merged_user.loc[
            group2_mask, 'centralized_loss_surrogate']).sum()

        # Group 3: 仅 video_id 匹配（user_id 不匹配）
        merged_video = pd.merge(
            train_selected,
            surrogate_selected,
            on='video_id',
            how='inner',
            suffixes=('_train', '_surrogate')
        )
        group3_mask = merged_video['user_id_train'] != merged_video['user_id_surrogate']
        group3_cov = (merged_video.loc[group3_mask, 'centralized_loss_train'] * merged_video.loc[
            group3_mask, 'centralized_loss_surrogate']).sum()

        # Group 4: 完全不匹配（可忽略或根据需求计算）
        # 此处假设不需要 Group4 的协方差贡献（如需要，可以通过全外连接筛选未匹配项）

        covariance_value = group1_cov+group2_cov+group3_cov

        # 确保 y_hat_train 和 y_hat_surrogate 的形状与对应的标签一致
        mse_loss_train = mse_loss_fn(y_hat_train.view(-1), train_labels.view(-1))  # 将形状调整为一维
        mse_loss_surrogate = mse_loss_fn(y_hat_surrogate.view(-1), surrogate_labels.view(-1))  # 将形状调整为一维

        # len_loss_train = len(mse_loss_train)
        # len_loss_surrogate = len(mse_loss_surrogate)

        mean_mse_train = mse_loss_train.mean().item()
        mean_mse_surrogate = mse_loss_surrogate.mean().item()
        var_mse_train = mse_loss_train.var().item()
        var_mse_surrogate = mse_loss_surrogate.var().item()  ## batch_size 太小会导致后面的批次循环缺失无偏数据

        mean_shift = mean_mse_train - mean_mse_surrogate

        var_bias_shift = torch.tensor(
            (1/len(mse_loss_train))*(var_mse_train ) +(1/len(mse_loss_surrogate))* (
                    var_mse_surrogate ) - 2*(covariance_value/(len(mse_loss_train)*len(mse_loss_surrogate))) 
                    )
        

        quantile_95 = norm.ppf(0.975) #norm.ppf(0.95)

        Upper_coef = quantile_95 * torch.sqrt(var_bias_shift) + abs(mean_shift)

        # 计算最优权重 best_alpha
        weight_frac = (1/len(mse_loss_surrogate))*(var_mse_surrogate) -  (covariance_value/(len(mse_loss_train)*len(mse_loss_surrogate)))
        weight_doma = (1/len(mse_loss_train))*(var_mse_train ) +(1/len(mse_loss_surrogate))* (
                    var_mse_surrogate ) + Upper_coef ** 2 - 2*(covariance_value/(len(mse_loss_train)*len(mse_loss_surrogate)))
        
        best_alpha = weight_frac / weight_doma

        # 确保 best_alpha 在 0 到 1 之间
        best_alpha = max(0.0, min(best_alpha, 1.0))
        # 记录当前 batch 的 best_alpha
        best_alpha_list.append(best_alpha)  # 将权重记录到当前 epoch 的列表中

        # 计算总损失
        total_loss = best_alpha * loss_train + (1 - best_alpha) * loss_surrogate

        total_loss.backward()
        optimizer.step()

        current_batch_size= max(len(train_labels), len(surrogate_labels))
        total_loss_epoch += total_loss.item() * current_batch_size
        total_batches += current_batch_size

    # 将当前 epoch 的 best_alpha_list 添加到矩阵中
    best_alpha_matrix.append(np.mean(torch.tensor(best_alpha_list).cpu().numpy()))

    epoch_mean_best_alpha = np.mean(torch.tensor(best_alpha_list).cpu().numpy())

    # 计算平均损失
    avg_train_loss = total_loss_epoch / total_batches

    loss_val, val_auc, _, _, _, _ = get_metrics(val_loader, model, k=5, device=device)

    avg_test_loss, test_auc, test_precision, test_recall, test_f1, test_ndcg_at_k = get_metrics(test_loader, model, k=5,
                                                                                                device=device)

    # 记录每个 epoch 的指标
    auc_score_list.append(test_auc)
    precision_score_list.append(test_precision)
    recall_score_list.append(test_recall)
    f1_score_list.append(test_f1)
    NDCG_score_list.append(test_ndcg_at_k)
    epoch_list.append(epoch + 1)
    train_loss_list.append(avg_train_loss)  # 保存测试集的平均损失
    test_loss_list.append(avg_test_loss)  # 保存测试集的平均损失
    # 调整学习率
    scheduler.step(val_auc)  # 关键步骤，ReduceLROnPlateau根据验证AUC调整学习率

    # 打印当前 epoch 的评估结果
    print(
        f'Epoch {epoch}/{epochs}: Train Loss: {avg_train_loss:.3f}, Test Loss: {avg_test_loss :.3f}, Val AUC: {val_auc:.3f},Test AUC: {test_auc:.3f}, Epoch_mean_best_alpha:{epoch_mean_best_alpha:.3f}, '
        f'Precision: {test_precision:.3f}, Recall: {test_recall:.3f}, F1 Score: {test_f1:.3f}, '
        f'NDCG@5: {test_ndcg_at_k:.3f}')
    # 调用早停逻辑
    early_stopping(val_auc, model)
    if early_stopping.early_stop:
        print("Early stopping triggered at epoch: {}".format(epoch))
        break

# Save metrics to CSV
metrics_df = pd.DataFrame({
    'Epoch': epoch_list,
    'Train Loss': train_loss_list,
    'Test Loss': test_loss_list,
    'AUC': auc_score_list,
    'Precision': precision_score_list,
    'Recall': recall_score_list,
    'F1 Score': f1_score_list,
    'NDCG@5': NDCG_score_list
})

# 将最佳权重矩阵保存到文件
np.savetxt("best_Loss_DC_Pessi_fintuning_alpha_matrix_tuning_{}_percent.txt".format(int(tuning_percentage * 100)),
           best_alpha_matrix, delimiter=",")

metrics_df.to_csv('metrics_trained_Loss_DC_Pessi_fintuning_{}_percent.csv'.format(int(tuning_percentage * 100)),
                  index=False)  # 依赖于微调数据比例
print("Metrics saved to metrics.csv")

# 记录结束时间
end_time = time.time()

# 打印运行时间
print(f"运行时间: {end_time - start_time:.2f} 秒")

# mlx worker launch -- python3 Loss_DC_Pessi_tuning.py