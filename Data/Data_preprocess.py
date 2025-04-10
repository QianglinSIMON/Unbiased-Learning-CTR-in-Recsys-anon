import numpy as np
import pandas as pd


dt_log_standard1 = pd.read_csv("/Users/bytedance/Desktop/Unbiased_learning_CTR_codes/kuai_rand_Datasets_preprocess/KuaiRand-Pure/data/log_standard_4_22_to_5_08_pure.csv")

dt_log_standard2 = pd.read_csv("/Users/bytedance/Desktop/Unbiased_learning_CTR_codes/kuai_rand_Datasets_preprocess/KuaiRand-Pure/data/log_standard_4_08_to_4_21_pure.csv")


dt_pure_log_standard=pd.concat([dt_log_standard1,dt_log_standard2],axis=0).reindex()

dt_pure_log_rand = pd.read_csv("/Users/bytedance/Desktop/Unbiased_learning_CTR_codes/kuai_rand_Datasets_preprocess/KuaiRand-Pure/data/log_random_4_22_to_5_08_pure.csv")



dt_pure_log_standard=dt_pure_log_standard[['user_id', 'video_id', 'time_ms', 'is_click', 'tab']] ## 选择主要指标和一个反馈指标

dt_pure_log_rand=dt_pure_log_rand[['user_id', 'video_id', 'time_ms', 'is_click', 'tab']] ## 选择主要指标和一个反馈指标

## 反馈指标：'is_click',
       # 'is_like', 'is_follow', 'is_comment', 'is_forward', 'is_hate',
       # 'long_view', 'play_time_ms', 'duration_ms', 'profile_stay_time',
       # 'comment_stay_time', 'is_profile_enter',




dt_pure_user_feature = pd.read_csv("/Users/bytedance/Desktop/Unbiased_learning_CTR_codes/kuai_rand_Datasets_preprocess/KuaiRand-Pure/data"
                                "/user_features_pure.csv")


dt_pure_user_feature=dt_pure_user_feature[['user_id', 'user_active_degree', 'is_lowactive_period',
       'is_live_streamer', 'is_video_author', 'follow_user_num',
       'follow_user_num_range', 'fans_user_num', 'fans_user_num_range',
       'friend_user_num', 'friend_user_num_range', 'register_days',
       'register_days_range', 'onehot_feat0', 'onehot_feat1', 'onehot_feat2',
       'onehot_feat3',  'onehot_feat5', 'onehot_feat6',
       'onehot_feat7', 'onehot_feat8', 'onehot_feat9', 'onehot_feat10',
       'onehot_feat11']] ## 删除 'onehot_feat4'，'onehot_feat12'，'onehot_feat13'，'onehot_feat14'，'onehot_feat15'，'onehot_feat16'，'onehot_feat17'，因为某些 user 的这几个指标为空。



dt_pure_video_basic_feature = pd.read_csv("/Users/bytedance/Desktop/Unbiased_learning_CTR_codes/kuai_rand_Datasets_preprocess/KuaiRand-Pure/data"
                                "/video_features_basic_pure.csv")


dt_pure_video_basic_feature=dt_pure_video_basic_feature[['video_id', 'author_id', 'video_type', 'upload_dt', 'upload_type',
       'visible_status',  'server_width', 'server_height',
       'music_id']] ## 删除 'music_type', 'tag'，'video_duration' 三列，因为有些视频的这三个特征为空 

dt_pure_video_summary_features = pd.read_csv("/Users/bytedance/Desktop/Unbiased_learning_CTR_codes/kuai_rand_Datasets_preprocess/KuaiRand-Pure/data/video_features_statistic_pure.csv")


# 将用户特征数据连接到推荐系统数据集中
##-- standard data---

pure_merged_standard_data1 = pd.merge(dt_pure_log_standard, dt_pure_user_feature, on='user_id', how='left')

pure_merged_standard_data2=pd.merge(pure_merged_standard_data1,dt_pure_video_basic_feature,on='video_id', how='left')

pure_merged_standard_data3=pd.merge(pure_merged_standard_data2,dt_pure_video_summary_features,on='video_id', how='left')

pure_merged_standard_data3.to_csv('normal_all_data.csv',index=False)

##-- random data---

pure_merged_rand_data1 = pd.merge(dt_pure_log_rand, dt_pure_user_feature, on='user_id', how='left')

pure_merged_rand_data2=pd.merge(pure_merged_rand_data1,dt_pure_video_basic_feature,on='video_id', how='left')

pure_merged_rand_data3=pd.merge(pure_merged_rand_data2,dt_pure_video_summary_features,on='video_id', how='left')

pure_merged_rand_data3.to_csv('random_all_data.csv',index=False)




