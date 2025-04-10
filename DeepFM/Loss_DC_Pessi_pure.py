import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score, precision_score, recall_score, f1_score, ndcg_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import sys
import os
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from deepfm import deepfm
from scipy.stats import norm

import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Data_preprocess_split import generate_data, get_metrics, custom_loss_function, EarlyStopping
import time
from itertools import cycle

# 记录开始时间
start_time = time.time()

target = ['is_click']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置超参数
batch_size = 1024
lr = 1e-4
wd = 0 #手动计算正则（需关闭优化器的weight_decay）
epochs = 500
l2_reg_linear = 1e-3
l2_reg_embedding = 5e-4
l2_reg_dnn = 5e-4
dnn_dropout = 0.8
embedding_size = 32
patience = 20
dnn_hidden_units = [400,400,400]
seed = 2024


torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
np.random.seed(seed)
random.seed(seed)

tuning_percentage = 0.25  # 0.05,0.1,0.15,0.2,0.25

batch_size_train = batch_size_surrogate = 128
# 调用 generate_data 函数，获取数据集和微调比例
df_biased, df_uniform, df_test, df_val, df_uniform_train, sparse_features, dense_features, feat_sizes, feature_names = generate_data(
    seed=seed)

# 随机抽取一定百分比的无偏数据用于微调
sample_size = int(len(df_uniform_train) * tuning_percentage)
df_tuning = df_uniform_train.sample(n=sample_size, replace=False, random_state=seed)

##--测试数据集加载器--
test_label = pd.DataFrame(df_test[target])
test_data = df_test.drop(columns=target[0])

test_tensor_data = torch.utils.data.TensorDataset(torch.from_numpy(np.array(test_data)),
                                                  torch.from_numpy(np.array(test_label)))

test_loader = DataLoader(dataset=test_tensor_data, shuffle=False, batch_size=batch_size)

##--验证数据集加载器--
val_label = pd.DataFrame(df_val[target])
val_data = df_val.drop(columns=target[0])

val_tensor_data = torch.utils.data.TensorDataset(torch.from_numpy(np.array(val_data)),
                                                 torch.from_numpy(np.array(val_label)))

val_loader = DataLoader(dataset=val_tensor_data, shuffle=False, batch_size=batch_size)

# 设定目标批次数
# target_num_batches = 15

# # 根据目标批次数计算批次大小
# batch_size_train = int(len(df_biased) / target_num_batches)
# batch_size_surrogate = int(len(df_tuning) / target_num_batches)


# --训练数据集--
train_label = pd.DataFrame(df_biased[target])
train_data = df_biased.drop(columns=[target[0]])

train_tensor_data = torch.utils.data.TensorDataset(torch.from_numpy(np.array(train_data)),
                                                   torch.from_numpy(np.array(train_label)))
train_loader = DataLoader(dataset=train_tensor_data, shuffle=True, batch_size=batch_size_train)

# --训练数据集--
df_labeled = df_tuning.reset_index(drop=True)

surrogate_label = pd.DataFrame(df_labeled[target])
surrogate_data = df_labeled.drop(columns=[target[0]])

surrogate_tensor_data = torch.utils.data.TensorDataset(torch.from_numpy(np.array(surrogate_data)),
                                                       torch.from_numpy(np.array(surrogate_label)))
surrogate_loader = DataLoader(dataset=surrogate_tensor_data, shuffle=True, batch_size=batch_size_surrogate)

auc_score_list = []
precision_score_list = []
recall_score_list = []
f1_score_list = []
NDCG_score_list = []
epoch_list = []
train_loss_list = []  # 保存训练集上的平均损失
test_loss_list = []  # 保存测试集上的平均损失

# 初始化模型和优化器
model = deepfm(feat_sizes, sparse_feature_columns=sparse_features, dense_feature_columns=dense_features,
               dnn_hidden_units=dnn_hidden_units, dnn_dropout=dnn_dropout, embedding_size=embedding_size,
               l2_reg_linear=l2_reg_linear, l2_reg_embedding=l2_reg_embedding, l2_reg_dnn=l2_reg_dnn,
               init_std=0.0001, seed=seed,
               device=device)


optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

mse_loss_fn = nn.BCELoss(reduction='none')  # 使用 MSE 损失计算
# 初始化早停
early_stopping = EarlyStopping(patience=patience, min_delta=0.001,
                               save_path="Loss_DC_Pessi_model_pure_{}_percent.pth".format(
                                   int(100 * tuning_percentage)))

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

    total_loss_epoch = 0.0  # 初始化每个 epoch 的总损失
    total_batches = 0  # 统计批次数
    best_alpha_list = []  # 当前 epoch 的 batch 权重列表

    # 使用 zip 并考虑较短的加载器
    for (x_train, y_train), (x_surrogate, y_surrogate) in zip(train_loader, surrogate_loader):
        optimizer.zero_grad()

        # 处理训练集数据
        x_train = x_train.to(device).float()
        y_train = y_train.to(device).float().unsqueeze(1).squeeze(-1)  # 调整形状
        y_hat_train = model(x_train)
        loss_train = custom_loss_function(y_hat_train, y_train, model,
                                          l2_reg_linear=model.l2_reg_linear,
                                          l2_reg_embedding=model.l2_reg_embedding,
                                          l2_reg_dnn=model.l2_reg_dnn)

        # 处理代理数据
        x_surrogate = x_surrogate.to(device).float()
        y_surrogate = y_surrogate.to(device).float().unsqueeze(1).squeeze(-1)  # 调整形状
        y_hat_surrogate = model(x_surrogate)
        loss_surrogate = custom_loss_function(y_hat_surrogate, y_surrogate, model,
                                              l2_reg_linear=model.l2_reg_linear,
                                              l2_reg_embedding=model.l2_reg_embedding,
                                              l2_reg_dnn=model.l2_reg_dnn)

        # 将 x_train 和 y_train 合并为一个数据集
        train_batch_data = torch.cat((x_train, y_train), dim=1)

        # 使用 feature_names 和 label 列名构造数据集
        columns = feature_names + target
        train_batch_df = pd.DataFrame(train_batch_data.cpu().numpy(), columns=columns)
        # 将 x_surrogate 和 y_surrogate 合并为一个数据集
        surrogate_batch_data = torch.cat((x_surrogate, y_surrogate), dim=1)

        # 使用 feature_names 和 label 列名构造数据集
        columns = feature_names + target
        surrogate_batch_df = pd.DataFrame(surrogate_batch_data.cpu().numpy(), columns=columns)

        # 为数据集添加来源标签
        train_batch_df['source'] = 'biased'
        surrogate_batch_df['source'] = 'unbiased'

        # 计算有偏数据集的损失值
        data_loss_train = mse_loss_fn(y_hat_train, y_train).view(-1)  # 确保为一维
        train_batch_df['loss'] = data_loss_train.detach().cpu().numpy()  # 将损失值添加到有偏数据集

        # 计算无偏数据集的损失值
        data_loss_surrogate = mse_loss_fn(y_hat_surrogate, y_surrogate).view(-1)  # 确保为一维
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

        # 基于损失函数计算均方损失的均值、方差和协方差
        mse_loss_train = mse_loss_fn(y_hat_train, y_train).view(-1)  # 确保为一维
        mse_loss_surrogate = mse_loss_fn(y_hat_surrogate, y_surrogate).view(-1)  # + reg_loss_val # 确保为一维

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

        # cauchy_value =(1 / len(mse_loss_train)) * (var_mse_train) + (1 / len(mse_loss_surrogate)) * (
        #     var_mse_surrogate) - 2 * (covariance_value / (len(mse_loss_train) * len(mse_loss_surrogate)))



        # 确保 best_alpha 在 0 到 1 之间
        best_alpha = max(0.0, min(best_alpha, 1.0))
        # 记录当前 batch 的 best_alpha
        best_alpha_list.append(best_alpha)  # 将权重记录到当前 epoch 的列表中

        # 计算总损失
        total_loss = best_alpha * loss_train + (1 - best_alpha) * loss_surrogate

        current_batch_size= max(len(y_train), len(y_surrogate))
        total_loss_epoch += total_loss.item() * current_batch_size
        total_batches += current_batch_size


        total_loss.backward()
        optimizer.step()

    # 将当前 epoch 的 best_alpha_list 添加到矩阵中
    best_alpha_matrix.append(best_alpha_list)

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
np.savetxt("DF_best_Pessi_pure_{}_percent_alpha_matrix.txt".format(int(100 * tuning_percentage)),
           best_alpha_matrix,
           delimiter=",")

metrics_df.to_csv('metrics_trained_LOSS_DC_Pessi_pure_{}_percent.csv'.format(int(100 * tuning_percentage)),
                  index=False)  # 依赖于微调数据比例
print("Metrics saved to metrics.csv")

# 记录结束时间
end_time = time.time()

# 打印运行时间
print(f"运行时间: {end_time - start_time:.2f} 秒")

# mlx worker launch -- python3 Loss_DC_Pessi_pure.py 

