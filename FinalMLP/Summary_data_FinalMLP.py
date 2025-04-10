import pandas as pd

# 数据准备
data_pretrained = {
    'auc': (
        [0.693, 0.696,0.697, 0.695,0.694] +
        [0.706,0.706,0.705,0.704,0.704] +
        [0.716,0.715,0.715,0.713,0.714] +
        [0.720,0.719,0.719,0.719,0.719] +
        [0.726,0.726,0.727,0.726,0.725]+
        [0.732,0.732,0.732,0.731,0.732]
    ),
    'test_loss':(
    [0.801, 0.801,0.801,0.816,0.775]+
    [0.512,0.494,0.487,0.496,0.482]+
    [0.510,0.491,0.484,0.494,0.478]+
    [0.513,0.496,0.478,0.487,0.483]+
    [0.503,0.484,0.476,0.485,0.471]+
    [0.500,0.483,0.474,0.484,0.469]
    ),
    'method': ['Pretrained_Tuning'] * 30,
    'percentage': (
        ['0%'] * 5 +
        ['5%'] * 5 +
        ['10%'] * 5 +
        ['15%'] * 5 +
        ['20%'] * 5+
        ['25%'] * 5
    )
}

# 转换为数据框
df_pretrained = pd.DataFrame(data_pretrained)

# 数据准备
data_Nonpessi_tuning = {
    'auc': (
   [0.708,0.707,0.709,0.707,0.706]+
   [0.717,0.717,0.718,0.716,0.716]+
   [0.723,0.722,0.722,0.721,0.722]+
   [0.727,0.728,0.727,0.726,0.728]+
   [0.731,0.732,0.733,0.732,0.732]
    ),
    'test_loss':(
[0.519,0.507,0.493,0.506,0.486]+
[0.519,0.499,0.488,0.501,0.485]+
[0.518,0.508,0.491,0.499,0.480]+
[0.521,0.511,0.485,0.495,0.485]+
[0.519,0.498,0.479,0.492,0.477]
    ),
    "weights_est":(
    [0.022,0.022,0.019,0.021,0.017]+
  [0.024,0.023,0.019,0.026,0.020]+
  [0.021,0.020,0.016,0.019,0.016]+
  [0.018,0.020,0.025,0.029,0.015]+
  [0.017,0.031,0.024,0.026,0.026]
    ),
    'method': ['Loss_DC_Nonpessi_Tuning'] * 25,
    'percentage': (
        ['5%'] * 5 +
        ['10%'] * 5 +
        ['15%'] * 5 +
        ['20%'] * 5+
        ['25%'] * 5
    )
}

# 转换为数据框
df_Nonpessi_tuning= pd.DataFrame(data_Nonpessi_tuning)

# 数据准备
data_Pessi_tuning = {
    'auc': (
   [0.707,0.707,0.708,0.707,0.705]+
   [0.716,0.716,0.717,0.715,0.715]+
   [0.722,0.721,0.721,0.721,0.721]+
   [0.726,0.727,0.727,0.726,0.727]+
   [0.731,0.732,0.733,0.732,0.732]
    ),
    'test_loss':(
   [0.515,0.502,0.490,0.502,0.484]+
   [0.514,0.494,0.485,0.497,0.481]+
   [0.513,0.500,0.487,0.496,0.478]+
   [0.514,0.501,0.481,0.490,0.480]+
   [0.506,0.490,0.476,0.487,0.473]
    ),
    "weights_est":(
   [0.009,0.009,0.008,0.009,0.008]+
   [0.009,0.009,0.008,0.009,0.008]+
   [0.008,0.008,0.007,0.007,0.007]+
   [0.007,0.007,0.010,0.010,0.006]+
   [0.010,0.010,0.009,0.010,0.009]
    ),
    'method': ['Loss_DC_Pessi_Tuning'] * 25,
    'percentage': (
         ['5%'] * 5 +
        ['10%'] * 5 +
        ['15%'] * 5 +
        ['20%'] * 5+
        ['25%'] * 5
    )
}

# 转换为数据框
df_Pessi_tuning= pd.DataFrame(data_Pessi_tuning)

# 数据准备
data_Supervised_all = {
          'auc': (
        [0.696,0.699,0.696,0.699,0.698] +
        [0.695,0.699,0.696,0.697,0.700] +
        [0.702,0.701,0.699,0.704,0.704] +
        [0.706,0.702,0.703,0.701,0.706] +
        [0.706,0.707,0.707,0.706,0.708]
    ),
    'test_loss':(
    [0.627,0.619,0.641,0.652,0.618]+
    [0.588,0.565,0.637,0.587,0.597]+
    [0.559,0.619,0.500,0.562,0.552]+
    [0.529,0.535,0.527,0.535,0.546]+
    [0.524,0.521,0.558,0.618,0.517]
    ),
    'method': ['Supervised_All'] * 25,
    'percentage': (
         ['5%'] * 5 +
        ['10%'] * 5 +
        ['15%'] * 5 +
        ['20%'] * 5+
        ['25%'] * 5
    )
}

df_Supervised_all= pd.DataFrame(data_Supervised_all)


# 数据准备
data_CV_Search_tuning = {
       'auc': (
        [0.706,0.707,0.707,0.704,0.706 ] +
        [0.716,0.717,0.716,0.714,0.714] +
        [0.719,0.721,0.720,0.719,0.718] +
        [0.724,0.724,0.722,0.722,0.720] +
        [0.731,0.730,0.731,0.728,0.729]
    ),
    'test_loss':(
    [0.651,0.631,0.623,0.627,0.618]+
    [0.644,0.653,0.659,0.647,0.607]+
    [0.632,0.648,0.622,0.649,0.627]+
    [0.653,0.625,0.619,0.644,0.606]+
    [0.607,0.603,0.610,0.621,0.607]
    ),
    "weights_search": (
            [0.8,0.8,0.8,0.8,0.8] +
            [0.8, 0.8, 0.8, 0.8, 0.8] +
            [0.8,0.8,0.8,0.8,0.8] +
            [0.8,0.8,0.8,0.8,0.8] +
            [0.6, 0.6, 0.6, 0.6, 0.6]
    ),
    'method': ['CV_Search_Tuning'] * 25,
    'percentage': (
         ['5%'] * 5 +
        ['10%'] * 5 +
        ['15%'] * 5 +
        ['20%'] * 5+
        ['25%'] * 5
    )
}

# 转换为数据框
data_CV_Search_tuning= pd.DataFrame(data_CV_Search_tuning)

# 数据准备
data_Only_tuning = {
          'auc': (
        [0.623,0.624,0.626,0.630,0.631] +
        [0.659,0.655,0.657,0.657,0.659] +
        [0.671,0.674,0.677,0.678,0.674] +
        [0.687,0.690,0.689,0.691,0.690] +
        [0.695,0.698,0.698,0.699,0.699]
    ),
    'test_loss':(
    [1.521,1.587,1.502,1.522,1.511]+
    [1.260,1.233,1.238,1.241,1.260]+
    [1.078,1.088,1.074,1.105,1.108,]+
    [0.952,0.952,0.973,0.955,0.971]+
    [0.896,0.882,0.985,0.932,0.933]
    ),
    'method': ['Only_Tuning'] * 25,
    'percentage': (
         ['5%'] * 5 +
        ['10%'] * 5 +
        ['15%'] * 5 +
        ['20%'] * 5+
        ['25%'] * 5
    )
}

# 转换为数据框
df_Only_tuning= pd.DataFrame(data_Only_tuning)


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 合并数据集
df_all = pd.concat([df_pretrained, df_Nonpessi_tuning, df_Pessi_tuning, data_CV_Search_tuning, df_Only_tuning,df_Supervised_all])
df_all.to_csv("df_all_summary_FinalMLP.csv", index=False)


#
# # 定义方法顺序和百分比顺序
# method_order = ['Supervised_all','Only_tuning', 'Scalinglaw_tuning', 'Pretrained', 'Pessi_tuning', 'Nonpessi_tuning']
# percentage_order = ['0%', '5%', '10%', '15%', '20%', '25%']
#
# # 将 percentage 列转换为有序分类类型
# df_all['percentage'] = pd.Categorical(df_all['percentage'], categories=percentage_order, ordered=True)
#
# # 绘制 AUC 箱线图
# plt.figure(figsize=(10, 6))
# sns.boxplot(data=df_all, x='percentage', y='auc', hue='method', hue_order=method_order)
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)  # 图例调整到下方
# plt.title("AUC Distribution by Percentage and Method")
# plt.tight_layout()
# plt.savefig("auc_boxplot.png")
# plt.show()
#
# # 按百分比-方法分组求 AUC 和测试集损失的均值
# df_avg = df_all.groupby(['method', 'percentage'], observed=False, as_index=False)[['auc', 'test_loss']].mean()
# df_avg.to_csv("df_auc_summary_deepfm.csv", index=False)
#
# # 绘制 AUC 的轨迹图
# plt.figure(figsize=(10, 6))
# sns.lineplot(data=df_avg, x='percentage', y='auc', hue='method', hue_order=method_order, marker='o')
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
# plt.title("Average AUC by Percentage and Method")
# plt.tight_layout()
# plt.savefig("auc_lineplot.png")
# plt.show()
#
# # 绘制测试集损失的轨迹图
# plt.figure(figsize=(10, 6))
# sns.lineplot(data=df_avg, x='percentage', y='test_loss', hue='method', hue_order=method_order, marker='o')
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
# plt.title("Average Test Loss by Percentage and Method")
# plt.tight_layout()
# plt.savefig("test_loss_lineplot.png")
# plt.show()
#
# # 处理 Nonpessi 和 Pessi 的权重数据
# df_weights = df_all[df_all['method'].isin(['Nonpessi_tuning', 'Pessi_tuning'])].copy()
#
# # 检查是否存在 weights_est 列
# if 'weights_est' in df_weights.columns:
#     # 按方法和百分比分组求权重均值
#     df_weights_avg = df_weights.groupby(['method', 'percentage'], observed=False, as_index=False)['weights_est'].mean()
#
#     # 绘制 Nonpessi 和 Pessi 权重变化的轨迹图
#     plt.figure(figsize=(10, 6))
#     sns.lineplot(data=df_weights_avg, x='percentage', y='weights_est', hue='method', marker='o')
#     plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
#     plt.title("Average Weights by Percentage (Nonpessi vs Pessi)")
#     plt.tight_layout()
#     plt.savefig("weights_lineplot.png")
#     plt.show()
# else:
#     print("No weights_est column found for Nonpessi and Pessi methods.")
