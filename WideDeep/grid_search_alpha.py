from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader
from Data_preprocess_split import get_metrics, custom_loss_function, EarlyStopping
import numpy as np
import pandas as pd
import torch.optim as optim
from Wide_deep_model import Wide_deep
from sklearn.model_selection import StratifiedKFold


def Grid_serch_alpha_CV(alpha_values, tuning_batch_size, df_unbiased, df_biased,
                        feat_sizes, sparse_feature_columns, dense_feature_columns,
                        dnn_hidden_units, dnn_dropout, embedding_size,
                        l2_reg_linear, l2_reg_embedding, l2_reg_dnn, 
                        patience, batch_size, lr, wd, epochs, seed, device):
    best_alpha = None
    best_val_auc = 0

    #kf = KFold(n_splits=3, shuffle=True, random_state=seed)
    
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    
    for alpha in alpha_values:
        print(f"Testing alpha = {alpha}")
        fold_val_aucs = []

        #for fold, (train_idx, val_idx) in enumerate(kf.split(df_unbiased)):
        

        # 使用目标变量进行分层
        for fold, (train_idx, val_idx) in enumerate(kf.split(df_unbiased, df_unbiased["is_click"])):
            
            print(f"Fold {fold + 1}")

            df_unbiased_train = df_unbiased.iloc[train_idx].reset_index(drop=True)
            df_unbiased_val = df_unbiased.iloc[val_idx].reset_index(drop=True)

            batch_size_train =  batch_size_surrogate = tuning_batch_size

            # 有偏数据集
            train_label = pd.DataFrame(df_biased['is_click'])
            train_data = df_biased.drop(columns=['is_click'])
            train_tensor_data = torch.utils.data.TensorDataset(
                torch.from_numpy(train_data.to_numpy().astype(np.float32)),
                torch.from_numpy(train_label.to_numpy().astype(np.float32))
            )
            train_loader = DataLoader(dataset=train_tensor_data, shuffle=True, batch_size=batch_size_train)

            # 无偏数据集
            surrogate_label = pd.DataFrame(df_unbiased_train['is_click'])
            surrogate_data = df_unbiased_train.drop(columns=['is_click'])
            surrogate_tensor_data = torch.utils.data.TensorDataset(
                torch.from_numpy(surrogate_data.to_numpy().astype(np.float32)),
                torch.from_numpy(surrogate_label.to_numpy().astype(np.float32))
            )
            surrogate_loader = DataLoader(dataset=surrogate_tensor_data, shuffle=True, batch_size=batch_size_surrogate)

            # 验证集
            val_label = pd.DataFrame(df_unbiased_val['is_click'])
            val_data = df_unbiased_val.drop(columns=['is_click'])
            val_tensor_data = torch.utils.data.TensorDataset(
                torch.from_numpy(val_data.to_numpy().astype(np.float32)),
                torch.from_numpy(val_label.to_numpy().astype(np.float32))
            )
            val_loader = DataLoader(dataset=val_tensor_data, shuffle=False, batch_size=batch_size)

            model_pre = Wide_deep(feat_sizes, 
            sparse_feature_columns=sparse_feature_columns, 
            dense_feature_columns=dense_feature_columns,
            dnn_hidden_units=dnn_hidden_units, 
            dnn_dropout=dnn_dropout, 
            embedding_size=embedding_size,
            l2_reg_linear=l2_reg_linear, 
            l2_reg_embedding=l2_reg_embedding, 
            l2_reg_dnn=l2_reg_dnn, 
            init_std=0.0001, 
            seed=seed,device=device).to(device)  # 确保模型参数在GPU上

            # 加载预训练的模型参数
            model_pre.load_state_dict(torch.load('pre_trained_model_{}.pth'.format(int(seed)), map_location=device, weights_only=False))

            optimizer_pre = optim.Adam(model_pre.parameters(), lr=lr, weight_decay=wd)
            early_stopping = EarlyStopping(patience=patience, min_delta=0.001,
                                           save_path=f"best_alpha_model_alpha_try.pth")

            # 训练模型
            
            for epoch in range(epochs):
                model_pre.train()
                
                for (x_train, y_train), (x_surrogate, y_surrogate) in zip(train_loader, surrogate_loader):
                    optimizer_pre.zero_grad()

                    x_train = x_train.to(device).float()
                    y_train = y_train.to(device).float().unsqueeze(1).squeeze(-1)
                    y_hat_train = model_pre(x_train)
                    loss_train = custom_loss_function(y_hat_train, y_train, model_pre)

                    x_surrogate = x_surrogate.to(device).float()
                    y_surrogate = y_surrogate.to(device).float().unsqueeze(1).squeeze(-1)
                    y_hat_surrogate = model_pre(x_surrogate)
                    loss_surrogate = custom_loss_function(y_hat_surrogate, y_surrogate, model_pre)

                    total_loss = alpha * loss_train + (1 - alpha) * loss_surrogate
                    total_loss.backward()
                    optimizer_pre.step()

                loss_val, val_auc, _, _, _, _ = get_metrics(val_loader, model_pre, k=5, device=device)
                print(f"Validation AUC for alpha = {alpha} at epoch {epoch}: {val_auc}")

                early_stopping(val_auc, model_pre)
                if early_stopping.early_stop:
                    print(f"Early stopping triggered for alpha = {alpha} at epoch {epoch}")
                    break

            fold_val_aucs.append(val_auc)

        mean_val_auc = np.mean(fold_val_aucs)
        print(f"Mean Validation AUC for alpha = {alpha}: {mean_val_auc}")

        if mean_val_auc > best_val_auc:
            best_val_auc = mean_val_auc
            best_alpha = alpha

    print(f"Best alpha: {best_alpha} with mean validation AUC: {best_val_auc}")
    
    return best_alpha, best_val_auc
