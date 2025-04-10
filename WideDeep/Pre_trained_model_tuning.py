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
from Wide_deep_model import Wide_deep


from torch.optim.lr_scheduler import ReduceLROnPlateau
from Data_preprocess_split import generate_data, get_metrics, custom_loss_function,EarlyStopping
import time 
# 记录开始时间
start_time = time.time()

target = ['is_click']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 设置超参数
batch_size = 1024
lr = 1e-5
wd = 0
epochs = 500
l2_reg_linear=1e-3
l2_reg_embedding=5e-4
l2_reg_dnn=5e-4
dnn_dropout=0.8
embedding_size=32
patience=20
dnn_hidden_units=[400,400,400]
seed = 2028



torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
np.random.seed(seed)
random.seed(seed)

tuning_percentage = 0.25 # 0.05, 0.1, 0.15, 0.2, 0.25 

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



tuning_batch_size=256

#--额外的无偏数据集--
tuning_uniform_label = pd.DataFrame(df_tuning[target]) #依赖于微调数据
tuning_uniform_data = df_tuning.drop(columns=[target[0]])


tuning_uniform_tensor_data = torch.utils.data.TensorDataset(torch.from_numpy(np.array(tuning_uniform_data)),
                                                   torch.from_numpy(np.array(tuning_uniform_label)))
tuning_uniform_loader = DataLoader(dataset=tuning_uniform_tensor_data, shuffle=True, batch_size=tuning_batch_size)

# 模型实例化
model = Wide_deep(feat_sizes, sparse_feature_columns=sparse_features, dense_feature_columns=dense_features,
               dnn_hidden_units=dnn_hidden_units, dnn_dropout=dnn_dropout, embedding_size=embedding_size,
               l2_reg_linear=l2_reg_linear, l2_reg_embedding=l2_reg_embedding, l2_reg_dnn=l2_reg_dnn, 
               init_std=0.0001, seed=seed,
               device=device)


# # 将模型转移到正确的设备上
# model.to(device)
# # 检查模型参数是否在 GPU 上
# assert next(model.parameters()).is_cuda, "Model parameters are not on the GPU!"

# 加载预训练的模型参数
model.load_state_dict(torch.load('pre_trained_model_{}.pth'.format(int(seed)), map_location=device, weights_only=False))

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

# # 冻结除了最后一层 self.dnn_outlayer、FM层和 bias 的其他参数
# for name, param in model.named_parameters():
#     if 'dnn_outlayer' not in name and 'fm' not in name and 'bias' not in name:
#         param.requires_grad = False

# # 只更新 self.dnn_outlayer 的参数
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)

# 检查模型参数是否在 GPU 上，并且确保参数冻结操作生效
# assert next(model.parameters()).is_cuda, "Model parameters are not on the GPU!"

auc_score_list = []
precision_score_list = []
recall_score_list = []
f1_score_list = []
NDCG_score_list = []
epoch_list = []
train_loss_list = []  # 保存训练集上的平均损失
test_loss_list = []  # 保存测试集上的平均损失


# 初始化早停
early_stopping = EarlyStopping(patience=patience, min_delta=0.001, save_path="pre_trained_model_tuning_{}_percentage.pth".format(int(tuning_percentage*100)))

for epoch in range(epochs):
    total_loss_epoch = 0.0
    total_tmp = 0

    model.train()
    for index, (x, y) in enumerate(tuning_uniform_loader):
        x = x.to(device).float()
        y = y.to(device).float()

        y_hat = model(x)

        optimizer.zero_grad()

        # 使用自定义损失函数
        loss = custom_loss_function(y_hat, y, model,
                                    l2_reg_linear=model.l2_reg_linear,
                                    l2_reg_embedding=model.l2_reg_embedding,
                                    l2_reg_dnn=model.l2_reg_dnn)
        loss.backward()
        optimizer.step()
        current_batch_size= y.size(0)
        total_loss_epoch += loss.item() * current_batch_size
        total_tmp += current_batch_size

    # 在验证集上计算AUC指标 用于早停机制
    loss_val, auc_val, _,_,_,_ = get_metrics(val_loader, model, k=5, device=device)

    avg_train_loss =total_loss_epoch / total_tmp
    # 在测试集上计算评估指标 
    avg_test_loss, auc, precision, recall, f1, ndcg_at_k = get_metrics(test_loader, model, k=5, device=device)
    auc_score_list.append(auc)
    precision_score_list.append(precision)
    recall_score_list.append(recall)
    f1_score_list.append(f1)
    NDCG_score_list.append(ndcg_at_k)
    epoch_list.append(epoch + 1)

    train_loss_list.append(avg_train_loss)  # 保存测试集的平均损失
    test_loss_list.append(avg_test_loss)  # 保存测试集的平均损失
    # 调整学习率
    scheduler.step(auc_val)  # 关键步骤，ReduceLROnPlateau根据验证AUC调整学习率

    print(
        'epoch/epochs: {}/{}, train loss: {:.3f}, test loss: {:.3f}, val auc: {:.3f}, test auc: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1_score: {:.3f}, ndcg@5: {:.3f}'.format(
            epoch, epochs, avg_train_loss, avg_test_loss, auc_val, auc, precision, recall, f1, ndcg_at_k))
    
    
    # 调用早停逻辑
    early_stopping(auc_val, model)
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

file_name = 'metrics_pretrained_model_tuning_{}_percentage.csv'.format(int(tuning_percentage*100)) ## 依赖于微调数据比例！
metrics_df.to_csv(file_name, index=False)
print("Metrics saved to metrics.csv")

# 记录结束时间
end_time = time.time()

# 打印运行时间
print(f"运行时间: {end_time - start_time:.2f} 秒")

# mlx worker launch -- python3 Pre_trained_model_tuning.py