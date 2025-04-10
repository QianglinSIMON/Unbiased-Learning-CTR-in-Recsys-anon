import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score,precision_score, recall_score, f1_score,ndcg_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler,  OrdinalEncoder,OneHotEncoder
import torch.nn as nn
import numpy as np


## 度量函数
def get_metrics(loader, model, k, device):
    pred, target = [], []
    model.eval()

    with torch.no_grad():
        total_test_loss = 0.0
        total_test_tmp = 0

        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).float()
            y_hat = model(x)
            pred.extend(y_hat.cpu().numpy().flatten())  # Ensure pred is 1D
            target.extend(y.cpu().numpy())

            # 使用自定义损失函数
            loss_test = custom_loss_function(y_hat, y, model,
                                    l2_reg_linear=model.l2_reg_linear,
                                    l2_reg_embedding=model.l2_reg_embedding,
                                    l2_reg_dnn=model.l2_reg_dnn)
            
            batch_size = y.size(0)
            total_test_loss += loss_test.item() * batch_size
            total_test_tmp += batch_size

        avg_test_loss = total_test_loss / total_test_tmp  # 计算测试集上的平均损失



    
    # Convert to numpy arrays
    pred = np.array(pred)
    target = np.array(target, dtype=int)  # Ensure target is integer type

    # Ensure target is binary and pred is probability
    assert all(t in [0, 1] for t in target), "Target values should be binary (0 or 1)"
    assert all(0 <= p <= 1 for p in pred), "Prediction values should be probabilities (between 0 and 1)"

    # Calculate AUC
    auc = roc_auc_score(target, pred)

    # Convert predictions to binary (0 or 1) based on a threshold, e.g., 0.5
    binary_pred = (pred >= 0.5).astype(int)

    # Calculate other metrics
    precision = precision_score(target, binary_pred, zero_division=0)
    recall = recall_score(target, binary_pred)
    f1 = f1_score(target, binary_pred)

    # 将 y_true 转换为适合 ndcg_score 的格式x`x`x`
    target_flat = target.flatten()
    y_true_multiclass = np.zeros((len(target_flat), 2))
    y_true_multiclass[np.arange(len(target_flat)), target_flat] = 1
    # unique_targets=np.unique(target)

    # 转换 y_score 以适应 ndcg_score
    y_score_multiclass = np.zeros((len(pred), 2))
    y_score_multiclass[:, 1] = pred
    y_score_multiclass[:, 0] = 1 - np.array(pred)

    # 计算 NDCG@k
    ndcg_at_k = ndcg_score(y_true_multiclass, y_score_multiclass, k=k)

    return avg_test_loss, auc, precision, recall, f1, ndcg_at_k


# 定义损失函数
def custom_loss_function(preds, targets, model, l2_reg_linear=0.001, l2_reg_embedding=0.0001, l2_reg_dnn=0.0001):
    # 计算主损失（例如二值交叉熵损失）
    main_loss = nn.BCELoss(reduction='mean')(preds, targets)

    # 获取正则化损失
    reg_loss = model.get_regularization_loss()

    # 将正则化损失加到主损失中
    total_loss = main_loss + reg_loss

    return total_loss


def extract_time_features(df, timestamp_col='time_ms'):
    dt = pd.to_datetime(df[timestamp_col], unit='ms')

    # 核心特征
    df['hour'] = dt.dt.hour
    df['weekday'] = dt.dt.weekday
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)

    # 周期编码
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # 业务时段
    df['is_peak'] = df['hour'].isin([9, 12, 19]).astype(int)

    return df.drop(timestamp_col, axis=1)


def generate_data(seed=2024):

    # 读取数据
    df_biased = pd.read_csv('/mlx_devbox/users/qianglin/playground/Datasets/normal_all_data.csv')
    df_uniform = pd.read_csv('/mlx_devbox/users/qianglin/playground/Datasets/random_all_data.csv')

    np.random.seed(seed)

    df_biased = extract_time_features(df_biased)
    df_uniform = extract_time_features(df_uniform)

    df_biased = df_biased.drop(columns=['visible_status', 'is_lowactive_period'])
    df_uniform = df_uniform.drop(columns=['visible_status', 'is_lowactive_period'])

    sparse_features = ['user_id', 'video_id', 'tab', 'hour', 'weekday', 'is_weekend', 'is_peak',
                       'user_active_degree', 'is_live_streamer',
                       'is_video_author', 'follow_user_num_x', 'follow_user_num_range',
                       'fans_user_num', 'fans_user_num_range', 'friend_user_num',
                       'friend_user_num_range', 'register_days', 'register_days_range',
                       'onehot_feat0', 'onehot_feat1', 'onehot_feat2', 'onehot_feat3',
                       'onehot_feat5', 'onehot_feat6', 'onehot_feat7', 'onehot_feat8',
                       'onehot_feat9', 'onehot_feat10', 'onehot_feat11', 'author_id',
                       'video_type', 'upload_dt', 'upload_type', 'music_id']

    dense_features = ['hour_sin', 'hour_cos',
                      'server_width', 'server_height', 'counts', 'show_cnt',
                      'show_user_num', 'play_cnt', 'play_user_num', 'play_duration',
                      'complete_play_cnt', 'complete_play_user_num', 'valid_play_cnt',
                      'valid_play_user_num', 'long_time_play_cnt', 'long_time_play_user_num',
                      'short_time_play_cnt', 'short_time_play_user_num', 'play_progress',
                      'comment_stay_duration', 'like_cnt', 'like_user_num', 'click_like_cnt',
                      'double_click_cnt', 'cancel_like_cnt', 'cancel_like_user_num',
                      'comment_cnt', 'comment_user_num', 'direct_comment_cnt',
                      'reply_comment_cnt', 'delete_comment_cnt', 'delete_comment_user_num',
                      'comment_like_cnt', 'comment_like_user_num', 'follow_cnt',
                      'follow_user_num_y', 'cancel_follow_cnt', 'cancel_follow_user_num',
                      'share_cnt', 'share_user_num', 'download_cnt', 'download_user_num',
                      'report_cnt', 'report_user_num', 'reduce_similar_cnt',
                      'reduce_similar_user_num', 'collect_cnt', 'collect_user_num',
                      'cancel_collect_cnt', 'cancel_collect_user_num',
                      'direct_comment_user_num', 'reply_comment_user_num', 'share_all_cnt',
                      'share_all_user_num', 'outsite_share_all_cnt']

    feature_names = dense_features + sparse_features


    # 标签特征
    new_cols = ['is_click'] + feature_names
    df_biased = df_biased[new_cols]
    df_uniform = df_uniform[new_cols]

    # 类别特征编码
    for feat in sparse_features:
        lbe = LabelEncoder()
        combined_data_item = pd.concat([df_biased[feat], df_uniform[feat]], axis=0)
        lbe.fit(combined_data_item)
        df_biased[feat] = lbe.transform(df_biased[feat])
        df_uniform[feat] = lbe.transform(df_uniform[feat])

    df_biased = df_biased.reset_index(drop=True)
    df_uniform = df_uniform.reset_index(drop=True)

    # 数值特征归一化
    mms = MinMaxScaler(feature_range=(0, 1))
    combined_data = pd.concat([df_biased, df_uniform], axis=0)
    combined_data[dense_features] = mms.fit_transform(
        combined_data[dense_features])  # fit_transform 组合了 fit 和 transform

    df_biased[dense_features] = mms.transform(df_biased[dense_features])
    df_uniform[dense_features] = mms.transform(df_uniform[dense_features])

    feat_size1 = {feat: 1 for feat in dense_features}
    ##--使用all data 上的所有类别特征的所有可能类别数
    feat_size2 = {feat: len(combined_data[feat].unique()) for feat in sparse_features}  ## 训练集上每个类别特征的所有可能类别数

    feat_sizes = {}
    feat_sizes.update(feat_size1)
    feat_sizes.update(feat_size2)

    # 切分无偏数据集
    total_size = len(df_uniform)
    # test_size = int(total_size * 0.2)
    test_size = int(total_size * 0.2)
    val_size = int(total_size * 0.2)

    # 随机打乱数据集
    df_uniform = df_uniform.sample(frac=1, random_state=seed).reset_index(drop=True)

    # 保证数据集之间互不重叠
    df_test = df_uniform.iloc[:test_size]
    df_val = df_uniform.iloc[test_size:test_size + val_size]
    df_uniform_train = df_uniform.iloc[test_size + val_size:]

    # 检查切分结果
    assert len(df_test) == test_size, "测试集大小不正确"
    assert len(df_val) == val_size, "验证集大小不正确"
    assert len(df_uniform_train) == (total_size - test_size - val_size), "训练集大小不正确"

    # df_biased :1436609*90; df_uniform: 1186059*90
    return df_biased, df_uniform, df_test, df_val, df_uniform_train, sparse_features, dense_features, feat_sizes, feature_names



class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, save_path="best_model.pth"):
        """
        :param patience: 允许验证集性能在多少轮次内没有提升。
        :param min_delta: 最小的改善幅度（严格大于该值才视为有改善）。
        :param save_path: 保存最佳模型的路径，默认保存为 'best_model.pth'。
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.save_path = save_path

    def __call__(self, val_score, model):
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score <= self.best_score + self.min_delta:
            # 如果 val_score 没有超过最佳分数至少 min_delta，计数器增加
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # 如果 val_score 提升幅度超过 min_delta，更新最佳分数并保存模型
            self.best_score = val_score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        """保存模型的最佳状态到指定路径"""
        torch.save(model.state_dict(), self.save_path)

    def load_best_model(self, model):
        """从保存的路径加载最佳模型"""
        model.load_state_dict(torch.load(self.save_path))