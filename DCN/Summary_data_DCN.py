import pandas as pd

# 数据准备
data_pretrained = {
    'auc': (
        [0.696,0.698,0.697,0.696,0.695] +
        [0.695,0.698,0.695, 0.696, 0.696] +
        [0.704,0.703, 0.710,0.706,0.704] +
        [0.713, 0.708, 0.720,0.713,0.711] +
        [0.719, 0.716,0.726,0.721,0.717]+
        [0.726, 0.724,0.731,0.728,0.725]
    ),
    'test_loss':(
    [0.761,0.698, 0.751,0.743,0.722]+
    [0.445,0.445,0.452,0.450, 0.448]+
    [0.444,0.443, 0.451,0.446,0.446]+
    [0.457, 0.442,0.443,0.443,0.446]+
    [0.447,0.442, 0.439,0.440,0.448]+
    [0.436,0.454,0.445,0.438,0.454]
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
   [0.702, 0.702,0.701,0.704,0.702]+
   [0.712, 0.708,0.711,0.714,0.711]+
   [0.718, 0.713,0.720, 0.718,0.717]+
   [0.726, 0.719,0.725,0.727,0.722]+
   [0.731,0.726,0.731,0.732,0.731]
    ),
    'test_loss':(
[0.445,0.445,0.452,0.448,0.448]+
[0.445,0.442,0.446,0.445,0.450]+
[0.439, 0.442,0.445,0.445,0.443]+
[0.442, 0.443,0.441,0.438, 0.442]+
[0.440,0.440,0.442,0.440,0.443]
    ),
    "weights_est":(
    [0.010,0.01,0.009,0.010,0.007]+
  [0.010,0.010,0.010, 0.009,0.008]+
  [0.012, 0.010,0.009,0.10,0.010]+
  [0.01,0.009,0.009,0.009,0.008]+
  [0.009,0.008,0.007,0.009,0.008]
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
   [0.700, 0.701,0.699,0.699,0.700]+
   [0.709, 0.706,0.712, 0.708,0.706]+
   [0.716, 0.712,0.723,0.713,0.713]+
   [0.724, 0.718, 0.727, 0.721,0.719]+
   [0.728, 0.725,0.734,0.729,0.725]
    ),
    'test_loss':(
   [0.445, 0.444,0.449, 0.449,0.448]+
   [0.447, 0.443,0.449,0.445,0.448]+
   [0.441,0.442,0.449,0.443,0.445]+
   [0.445,0.444,0.445,0.440, 0.445]+
   [0.443,0.441,0.441, 0.439,0.449]
    ),
    "weights_est":(
   [0.005, 0.005,0.005,0.005,0.004]+
   [0.009, 0.005,0.005,0.005,0.004]+
   [0.005,0.005,0.004,0.006,0.004]+
   [0.004,0.005,0.004,0.006,0.004]+
   [0.004,0.004,0.004,0.005,0.003]
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
        [0.703,0.705,0.703,0.702,0.705] +
        [0.705,0.707,0.704,0.707,0.704] +
        [0.706,0.708,0.706,0.709,0.708] +
        [0.710,0.710,0.708,0.712,0.708] +
        [0.709,0.710,0.709, 0.712,0.709]
    ),
    'test_loss':(
    [0.659, 0.627,0.630,0.614,0.608]+
    [0.619,0.587,0.585,0.580,0.594]+
    [0.564,0.560,0.540,0.556,0.560]+
    [0.524,0.533,0.560,0.523,0.534]+
    [0.556,0.496,0.530,0.520,0.527]
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
        [0.705, 0.708,0.705,0.707,0.706 ] +
        [0.710,0.712,0.709,0.712,0.711] +
        [0.713,0.715,0.712,0.715,0.714] +
        [0.715,0.717,0.715,0.717,0.716] +
        [0.722,0.722,0.723,0.725,0.725]
    ),
    'test_loss':(
    [0.540, 0.536,0.541,0.530,0.548]+
    [0.512,0.522,0.524,0.519, 0.510]+
    [0.505,0.506,0.509,0.510,0.511]+
    [0.506,0.508,0.509,0.507,0.504]+
    [0.463,0.466,0.472,0.475,0.463]
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
        [0.573, 0.607, 0.566,0.590,0.620] +
        [0.651,0.664,0.600,0.650,0.662] +
        [0.686, 0.686,0.680,0.684,0.686] +
        [0.701,0.703,0.698,0.701,0.704] +
        [0.711,0.714,0.710,0.711,0.714]
    ),
    'test_loss':(
    [0.965, 0.718, 0.637,0.902,0.791]+
    [0.601,0.594,0.831,0.619,0.610]+
    [0.543, 0.551,0.558,0.562,0.561]+
    [0.531,0.532,0.524,0.528,0.536]+
    [0.511,0.516,0.504,0.514,0.518]
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
df_all.to_csv("df_all_summary_DCN.csv", index=False)


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
