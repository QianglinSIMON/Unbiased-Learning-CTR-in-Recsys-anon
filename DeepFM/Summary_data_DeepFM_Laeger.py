import pandas as pd

# 数据准备
data_pretrained = {
    'auc': (
        [0.749,0.751,0.750,0.750,0.750]+
        [0.756,0.757,0.757,0.757,0.757] +
        [0.762,0.763,0.763,0.764,0.762]
    ),
    'test_loss':(
    [0.439,0.439,0.438,0.438,0.437]+
    [0.436,0.434,0.436,0.434,0.433]+
    [0.432,0.430,0.431,0.430,0.429]
    ),
    'method': ['Pretrained_Tuning'] * 15,
    'percentage': (
        ['50%'] * 5 +
        ['70%'] * 5+
        ['100%'] * 5
    )
}

# 转换为数据框
df_pretrained = pd.DataFrame(data_pretrained)

# 数据准备
data_Nonpessi_tuning = {
    'auc': (
   [0.749,0.751,0.750,0.750,0.751]+
   [0.757,0.758,0.757,0.759,0.757]+
   [0.762,0.763,0.763,0.764,0.762]
    ),
    'test_loss':(
[0.439,0.439,0.440,0.438,0.436]+
[0.436,0.434,0.436,0.434,0.433]+
[0.432,0.430,0.431,0.430,0.429]
    ),
    "weights_est":(
  [0.002,0.002,0.002,0.002,0.002]+
  [0.002,0.002,0.002,0.002,0.002]+
  [0.002,0.002,0.002,0.002,0.002]
    ),
    'method': ['Loss_DC_Nonpessi_Tuning'] * 15,
    'percentage': (
        ['50%'] * 5 +
        ['70%'] * 5+
        ['100%'] * 5
    )
}

# 转换为数据框
df_Nonpessi_tuning= pd.DataFrame(data_Nonpessi_tuning)

# 数据准备
data_Pessi_tuning = {
    'auc': (
   [0.749,0.751,0.750,0.750,0.751]+
   [0.756,0.757,0.757,0.759,0.757]+
   [0.762,0.763,0.763,0.764,0.762]
    ),
    'test_loss':(
   [0.439,0.439,0.440,0.438,0.436]+
   [0.436,0.434,0.436,0.434,0.433]+
   [0.432,0.430,0.431,0.430,0.429]
    ),
    "weights_est":(
   [0.001,0.001,0.001,0.001,0.001]+
   [0.001,0.001,0.001,0.001,0.001]+
   [0.001,0.001,0.001,0.001,0.001]
    ),
    'method': ['Loss_DC_Pessi_Tuning'] * 15,
    'percentage': (
        ['50%'] * 5 +
        ['70%'] * 5+
        ['100%'] * 5
    )
}

# 转换为数据框
df_Pessi_tuning= pd.DataFrame(data_Pessi_tuning)



# 数据准备
data_CV_Search_tuning = {
       'auc': (
        [0.745,0.745,0.745,0.745,0.745] +
        [0.751,0.751,0.750,0.751,0.750] +
        [0.755,0.756,0.755,0.757,0.756]
    ),
    'test_loss':(
    [0.443,0.442,0.444, 0.443,0.441]+
    [0.440,0.438,0.440,0.440,0.438]+
    [0.437,0.436,0.437,0.435,0.434]
    ),
    "weights_search": (
            [0.2,0.2,0.2,0.2,0.2] +
            [0.2,0.2,0.2,0.2,0.2] +
            [0.2,0.2,0.2,0.2,0.2]
    ),
    'method': ['CV_Search_Tuning'] * 15,
    'percentage': (
        ['50%'] * 5 +
        ['70%'] * 5+
        ['100%'] * 5
    )
}

# 转换为数据框
data_CV_Search_tuning= pd.DataFrame(data_CV_Search_tuning)

# 数据准备
data_Only_tuning = {
          'auc': (
        [0.738,0.737,0.736,0.739,0.740] +
        [0.745,0.745,0.744,0.750,0.746] +
        [0.750,0.752,0.750,0.759,0.753]
    ),
    'test_loss':(
    [0.456,0.455,0.456,0.457,0.453]+
    [0.452,0.441,0.452,0.444,0.451]+
    [0.449,0.447,0.448,0.432,0.447]
    ),
    'method': ['Only_Tuning'] * 15,
    'percentage': (
        ['50%'] * 5 +
        ['70%'] * 5+
        ['100%'] * 5
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
df_all = pd.concat([df_pretrained, df_Nonpessi_tuning, df_Pessi_tuning, data_CV_Search_tuning, df_Only_tuning])
df_all.to_csv("df_all_summary_DeepFM_Larger.csv", index=False)

