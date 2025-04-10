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
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Data_preprocess_split import generate_data, get_metrics, custom_loss_function,EarlyStopping

from sklearn.model_selection import KFold
from grid_search_alpha import Grid_serch_alpha_CV

import time
# 记录开始时间
start_time = time.time()


target = ['is_click']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置超参数
batch_size = 1024
lr = 1e-5
wd = 0 #手动计算正则（需关闭优化器的weight_decay）
epochs = 500
l2_reg_linear = 1e-3
l2_reg_embedding = 5e-4
l2_reg_dnn = 5e-4
dnn_dropout = 0.8
embedding_size = 32
patience = 20
dnn_hidden_units = [400,400,400]
seed = 2028


torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
np.random.seed(seed)
random.seed(seed)


tuning_percentage = 0.4 #0.05,0.1,0.15,0.2,0.25

batch_size_training= 1024
# 调用 generate_data 函数，获取数据集和微调比例
df_biased, df_uniform, df_test, df_val, df_uniform_train, sparse_features, dense_features, feat_sizes,feature_names = generate_data(seed=seed)

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




# 初始化最佳 alpha 和最佳验证 AUC
# best_alpha = None
# best_val_auc = 0

alpha_values = [0, 0.2, 0.4, 0.6, 0.8, 1]  # 定义一系列的 alpha 值




best_alpha, best_val_auc=Grid_serch_alpha_CV(alpha_values=alpha_values,batch_size_training=batch_size_training,
                                             df_unbiased=df_tuning,df_biased=df_biased,
                                             feat_sizes=feat_sizes, sparse_feature_columns=sparse_features, dense_feature_columns=dense_features,
                                dnn_hidden_units=dnn_hidden_units, dnn_dropout=dnn_dropout, embedding_size=embedding_size,
                                l2_reg_linear=l2_reg_linear, l2_reg_embedding=l2_reg_embedding, l2_reg_dnn=l2_reg_dnn, 
                                             patience=patience,batch_size=batch_size,
                                             lr=lr,wd=wd,epochs=epochs,seed=seed,device=device)


best_alpha=0.6 #(0.8,0.8,0.6,0.6,0.4)

# 根据目标批次数计算批次大小
batch_size_train = batch_size_surrogate_train = batch_size_training


#--训练数据集--
train_label = pd.DataFrame(df_biased['is_click'])
train_data = df_biased.drop(columns=['is_click'])

train_tensor_data = torch.utils.data.TensorDataset(torch.from_numpy(np.array(train_data)),
                                                   torch.from_numpy(np.array(train_label)))
train_loader = DataLoader(dataset=train_tensor_data, shuffle=True, batch_size=batch_size_train)



#--训练数据集--

df_labeled=df_tuning.reset_index(drop=True)

surrogate_label = pd.DataFrame(df_labeled['is_click'])
surrogate_data = df_labeled.drop(columns=['is_click'])

surrogate_tensor_data = torch.utils.data.TensorDataset(torch.from_numpy(np.array(surrogate_data)),
                                                   torch.from_numpy(np.array(surrogate_label)))
surrogate_loader = DataLoader(dataset=surrogate_tensor_data, shuffle=True, batch_size=batch_size_surrogate_train)



auc_score_list = []
precision_score_list = []
recall_score_list = []
f1_score_list = []
NDCG_score_list = []
epoch_list = []
train_loss_list = []  # 保存训练集上的平均损失
test_loss_list = []  # 保存测试集上的平均损失

# 重新在训练集上训练模型并在测试集上评估
model = deepfm(feat_sizes, sparse_feature_columns=sparse_features, dense_feature_columns=dense_features,
               dnn_hidden_units=dnn_hidden_units, dnn_dropout=dnn_dropout, embedding_size=embedding_size,
               l2_reg_linear=l2_reg_linear, l2_reg_embedding=l2_reg_embedding, l2_reg_dnn=l2_reg_dnn, 
               init_std=0.0001, seed=seed,
               device=device)

# 加载预训练的模型参数: 基于初始预训练模型
model.load_state_dict(torch.load('pre_trained_model_{}.pth'.format(int(seed)), map_location=device, weights_only=False))

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)


# 初始化早停
early_stopping = EarlyStopping(patience=patience, min_delta=0.001, save_path="Scaling_law_tuning_Pretrained_{}_percent.pth".format(int(100*tuning_percentage)))

for epoch in range(epochs):
    model.train()

    total_loss_epoch = 0.0  # 初始化总损失
    total_batches = 0  # 初始化批次数

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

        # 计算总损失
 
        total_loss = best_alpha*loss_train +(1- best_alpha) * loss_surrogate
        
        current_batch_size= max(len(y_train), len(y_surrogate))
        total_loss_epoch += total_loss.item() * current_batch_size
        total_batches += current_batch_size

        total_loss.backward()
        optimizer.step()

    # 计算平均损失
    avg_train_loss = total_loss_epoch / total_batches

    loss_val, val_auc, _,_,_,_ = get_metrics(val_loader, model, k=5,device=device)

    avg_test_loss, test_auc, test_precision, test_recall, test_f1, test_ndcg_at_k = get_metrics(test_loader, model, k=5,device=device)

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
    print(f'Epoch {epoch }/{epochs}: Train Loss: {avg_train_loss:.3f}, Test Loss: {avg_test_loss :.3f}, Val AUC: {val_auc:.3f},Test AUC: {test_auc:.3f}, '
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


metrics_df.to_csv('metrics_CV_search_tuning_Pretrained_{}_percent.csv'.format(int(100*tuning_percentage)), index=False) #依赖于微调数据比例
print("Metrics saved to metrics.csv")


# 记录结束时间
end_time = time.time()

# 打印运行时间
print(f"运行时间: {end_time - start_time:.2f} 秒")

# mlx worker launch -- python3 CV_search_finetuning.py


##-- bathc_size =128--
## 5%  Best alpha: 0.8 with mean validation AUC: 0.6915488026568074
## 10% Best alpha: 0.8 with mean validation AUC: 0.6998397451145563
## 15% Best alpha: 0.8 with mean validation AUC: 0.7051427926175057
## 20% Best alpha: 0.8 with mean validation AUC: 0.7104282386136839
## 25% Best alpha: 0.6 with mean validation AUC: 0.712370396917704

##-- bathc_size =256--
## 5% Best alpha: 0.8 with mean validation AUC: 0.6956194571087077
## 10% Best alpha: 0.8 with mean validation AUC: 0.701993857620311
## 15% Best alpha: 0.8 with mean validation AUC: 0.7062862454225508
## 20% Best alpha: 0.6 with mean validation AUC: 0.7109879849714699
## 25% Best alpha: 0.6 with mean validation AUC: 0.7138481401120872

##--batch_size =512--
## 5% Best alpha: 0.8 with mean validation AUC: 0.6988273400852211
## 10% Best alpha: 0.8 with mean validation AUC: 0.7040024838240658
## 15% Best alpha: 0.8 with mean validation AUC: 0.7071430768649778
## 20% Best alpha: 0.6 with mean validation AUC: 0.712726355824216
## 25% Best alpha: 0.6 with mean validation AUC: 0.7146571116471637

##--batch_size=1024--
## 5% Best alpha: 0.8 with mean validation AUC: 0.7010591111824004
## 10% Best alpha: 0.8 with mean validation AUC: 0.705220820905872
## 15% Best alpha: 0.6 with mean validation AUC: 0.7084125331270533
## 20% Best alpha: 0.6 with mean validation AUC: 0.7143524092171734
## 25% Best alpha: 0.4 with mean validation AUC: 0.7163973655057009

## 50% Best alpha: 0.2 with mean validation AUC: 0.7321376995911715
## 70% Best alpha: 0.2 with mean validation AUC: 0.7409196859062271
## 100% Best alpha: 0.2 with mean validation AUC: 0.7485209596067784