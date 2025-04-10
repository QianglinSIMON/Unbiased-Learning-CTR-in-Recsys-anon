import torch
from DCN_model import CDNet
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Data_preprocess_split import (generate_data, get_metrics, create_data_loaders,
                                   custom_loss_function, EarlyStopping)

import time

# 记录开始时间
start_time = time.time()

# 定义超参数
lr = 1e-4
batch_size = 1024
wd = 0
epochs = 500
cross_layer_num=3
dnn_hidden_units=[400, 400, 400]
l2_reg_dnn = 1e-3
l2_reg_embedding = 5e-4
l2_reg_dcn = 5e-4
patience = 20
dropout = 0.8
seed = 2028

torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
np.random.seed(seed)
random.seed(seed)

tuning_percentage = 0
# 调用 generate_data 函数，获取数据集和微调比例
df_biased, df_uniform, df_test, df_val, df_uniform_train, sparse_features, dense_features, feat_sizes, feature_names = generate_data(
    seed=seed)

# 随机抽取一定百分比的无偏数据用于微调
sample_size = int(len(df_uniform_train) * tuning_percentage)
# df_tuning = df_uniform_train.sample(n=sample_size, replace=False, random_state=seed)

sparse_feature_columns = sparse_features
dense_feature_columns = dense_features

# 获取稀疏特征的类别数和它们的索引
embedding_size = [feat_sizes[feat] for feat in sparse_features]
embedding_index= [sparse_features.index(feat) for feat in sparse_features]

target = ['is_click']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建数据加载器
train_loader = create_data_loaders(batch_size=batch_size, dataset=df_biased, sparse_features=sparse_features,
                                   dense_features=dense_features, target=target, shuffle=True)
test_loader = create_data_loaders(batch_size=batch_size, dataset=df_test, sparse_features=sparse_features,
                                  dense_features=dense_features, target=target, shuffle=False)
val_loader = create_data_loaders(batch_size=batch_size, dataset=df_val, sparse_features=sparse_features,
                                 dense_features=dense_features, target=target, shuffle=False)

# 创建模型实例
model = CDNet(embedding_index=embedding_index,embedding_size=embedding_size,
              dense_feature_num=len(dense_features),cross_layer_num=cross_layer_num,
              dnn_hidden_units=dnn_hidden_units,
              dropout_rate=dropout,
              l2_reg_embedding=l2_reg_embedding, l2_reg_dnn=l2_reg_dnn, l2_reg_dcn=l2_reg_dcn,
              device=device).to(device)

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

# 初始化早停
early_stopping = EarlyStopping(patience=patience, min_delta=0.001, save_path="pre_trained_model_{}.pth".format(int(seed)))

# 训练过程
for epoch in range(epochs):
    total_loss_epoch = 0.0
    total_tmp = 0

    model.train()
    for sparse_inputs, dense_inputs, labels in train_loader:
        sparse_inputs = sparse_inputs.to(device)
        dense_inputs = dense_inputs.to(device)
        y = labels.to(device).float()

        optimizer.zero_grad()
        y_hat = model(sparse_inputs, dense_inputs)
        y_hat = torch.sigmoid(y_hat)  # 确保输出在0和1之间

        # 使用自定义损失函数
        loss = custom_loss_function(y_hat, y, model,
                                    l2_reg_embedding=model.l2_reg_embedding,
                                    l2_reg_dnn=model.l2_reg_dnn,
                                    l2_reg_dcn=model.l2_reg_dcn
                                    )

        loss.backward()
        optimizer.step()

        current_batch_size= y.size(0)
        total_loss_epoch += loss.item() * current_batch_size
        total_tmp += current_batch_size

    # 验证集上的AUC计算用于早停
    loss_val, auc_val, _, _, _, _ = get_metrics(val_loader, model, k=5, device=device)
    avg_train_loss = total_loss_epoch / total_tmp

    # 测试集上的评估指标
    avg_test_loss, auc, precision, recall, f1, ndcg_at_k = get_metrics(test_loader, model, k=5, device=device)

    print(
        f'epoch/epochs: {epoch}/{epochs}, train loss: {avg_train_loss:.3f}, test loss: {avg_test_loss:.3f}, val auc: {auc_val:.3f}, test auc: {auc:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, f1_score: {f1:.3f}, ndcg@5: {ndcg_at_k:.3f}')

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

    # 早停逻辑
    early_stopping(auc_val, model)
    if early_stopping.early_stop:
        print(f"Early stopping triggered at epoch: {epoch}")
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

file_name = 'metrics_pretrained_model.csv'
metrics_df.to_csv(file_name, index=False)
print("Metrics saved to metrics_pretrained_model.csv")

# 记录结束时间
end_time = time.time()

# 打印运行时间
print(f"运行时间: {end_time - start_time:.2f} 秒")

# mlx worker launch -- python3 Pretrained_model.py

