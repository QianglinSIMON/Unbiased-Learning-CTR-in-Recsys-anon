from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import torch.optim as optim
from final_mlp import FinalMLP

from Data_preprocess_split import ( get_metrics, 
                                   create_data_loaders,
                                   custom_loss_function, EarlyStopping,get_sparse_emb_dims)



def Grid_serch_alpha_CV(alpha_values, tuning_batch_size, df_unbiased, df_biased,
                        sparse_features, dense_features, target,
                        feat_sizes,
                         dense_dim_embedding,
                        sparse_dim_embeddings,
                        sparse_output_dim,
                        dim_hidden_fs,
                        num_hidden_1,
                        dim_hidden_1,
                        num_hidden_2,
                        dim_hidden_2,
                        num_heads,dropout,
                        l2_reg_fs,
                        l2_reg_embedding,
                        l2_reg_mlp,
                        l2_reg_agg,seed,
                        lr, wd, epochs, patience, device
                        ):
    """
    使用交叉验证来搜索最优的alpha值
    """
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

            batch_size_train = batch_size_surrogate = tuning_batch_size

            # 有偏数据集
            train_label = pd.DataFrame(df_biased['is_click'])
            train_data = df_biased.drop(columns=['is_click'])
            train_tensor_data = torch.utils.data.TensorDataset(
                torch.from_numpy(train_data.to_numpy().astype(np.float32)),
                torch.from_numpy(train_label.to_numpy().astype(np.float32))
            )

            # 准备数据加载器
            train_loader = create_data_loaders(batch_size=batch_size_train, dataset=df_biased, sparse_features=sparse_features, dense_features=dense_features, target=target, shuffle=True)

            surrogate_loader = create_data_loaders(batch_size=batch_size_surrogate, dataset=df_unbiased_train, sparse_features=sparse_features, dense_features=dense_features, target=target, shuffle=True)


            val_loader = create_data_loaders(batch_size=batch_size_surrogate, dataset=df_unbiased_val, sparse_features=sparse_features, dense_features=dense_features, target=target, shuffle=False)

            model_pre =  FinalMLP(
    feat_sizes=feat_sizes,
    sparse_feature_columns=sparse_features,
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

            # 确保模型参数在GPU上

            # 加载预训练的模型参数
            model_pre.load_state_dict(torch.load('pre_trained_model_{}.pth'.format(int(seed)), map_location=device, weights_only=False))

            optimizer_pre = optim.Adam(model_pre.parameters(), lr=lr, weight_decay=wd)
            early_stopping = EarlyStopping(patience=patience, min_delta=0.001,
                                           save_path=f"best_alpha_model_alpha.pth")

            # 训练模型
            
            for epoch in range(epochs):

                model_pre.train()


                for (train_sparse_inputs_pre, train_dense_inputs_pre, train_labels_pre), (surrogate_sparse_inputs_pre, surrogate_dense_inputs_pre, surrogate_labels_pre) in zip(train_loader,surrogate_loader):
                    
                    optimizer_pre.zero_grad()

                    train_sparse_inputs_pre = train_sparse_inputs_pre.to(device)
                    train_dense_inputs_pre = train_dense_inputs_pre.to(device)
                    train_labels_pre = train_labels_pre.to(device).float()



                    y_hat_train_pre = model_pre(train_sparse_inputs_pre, train_dense_inputs_pre)

                    y_hat_train_pre=torch.sigmoid(y_hat_train_pre)  # 确保输出在0和1之间

                    # 使用自定义损失函数
                    loss_train_pre = custom_loss_function(y_hat_train_pre, train_labels_pre, model_pre,
                                                           l2_reg_mlp=model_pre.l2_reg_mlp,l2_reg_agg = model_pre.l2_reg_agg,
                                    l2_reg_fs = model_pre.l2_reg_fs,l2_reg_embedding = model_pre.l2_reg_embedding)

                    surrogate_sparse_inputs_pre= surrogate_sparse_inputs_pre.to(device)
                    surrogate_dense_inputs_pre= surrogate_dense_inputs_pre.to(device)
                    surrogate_labels_pre= surrogate_labels_pre.to(device).float()
                    y_hat_surrogate_pre = model_pre(surrogate_sparse_inputs_pre, surrogate_dense_inputs_pre)
                    y_hat_surrogate_pre = torch.sigmoid(y_hat_surrogate_pre)  # 确保输出在0和1之间
                    
                    loss_surrogate_pre = custom_loss_function(y_hat_surrogate_pre, surrogate_labels_pre, model_pre, 
                                                                 l2_reg_mlp=model_pre.l2_reg_mlp,l2_reg_agg = model_pre.l2_reg_agg,
                                    l2_reg_fs = model_pre.l2_reg_fs,l2_reg_embedding = model_pre.l2_reg_embedding)


                    # 计算总损失
                    total_loss_pre = alpha*loss_train_pre + (1-alpha) * loss_surrogate_pre

                    total_loss_pre.backward()
                    optimizer_pre.step()

                # 使用验证集来评估模型
                loss_val, val_auc, val_precision, val_recall, val_f1, val_ndcg_at_k = get_metrics(val_loader, model_pre, k=5,
                                                                                device=device)

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
