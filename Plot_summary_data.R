library(tidyverse)
library(ggplot2)
library(dplyr)
library(ggplot2)

analyze_and_plot <- function(dt_select, name) {
  
  # 参数校验
  if (!all(c("auc", "test_loss", "method", "percentage") %in% colnames(dt_select))) {
    stop("数据集必须包含 auc, test_loss, method, percentage 列")
  }
  
  # 1. 数据预处理（保持原始数据格式）
  method_levels <- c("Pretrained_Finetune", "Loss_DC_Nonpessi_Finetune",
                     "Loss_DC_Pessi_Finetune", "CV_Search_Finetune",
                     "Supervised_All", "Only_Unbiased")
  
  #percentage_levels <- c("0%", "5%", "10%", "15%", "20%", "25%")
  
  percentage_levels <- c("50%","70%","100%")
  
  dt_long <- dt_select %>%
    mutate(
      method = factor(method, levels = method_levels),
      percentage = factor(percentage, levels = percentage_levels)
    ) %>%
    pivot_longer(
      cols = c(auc, test_loss),
      names_to = "metric",
      values_to = "value"
    ) %>%
    mutate(
      metric = case_when(
        metric == "auc" ~ "AUC",
        metric == "test_loss" ~ "Test Loss",
        TRUE ~ metric
      )
    )
  
  # 2. 保存长格式数据
  write.csv(dt_long, paste0(name, "_long_format_data.csv"), row.names = FALSE)
  
  # 3. 创建蜂群图
  p_facet <- ggplot(dt_long, aes(x = percentage, y = value, color = method)) +
    # 蜂群图核心层
    ggbeeswarm::geom_beeswarm(
      dodge.width = 0.9,    # 分组间距
      size = 2.5,           # 点大小
      alpha = 0.8,          # 透明度
      method = "swarm",     # 排列方式
      cex = 0.8             # 点间距控制
    ) +
    # 分面显示指标
    facet_wrap(
      ~ metric,
      scales = "free_y",
      ncol = 2,
      labeller = labeller(metric = label_value)
    ) +
    # 颜色方案
    scale_color_manual(
      values = c(
        "Pretrained_Finetune" = "#E6194B",
        "Loss_DC_Nonpessi_Finetune" = "#3CB44B",
        "Loss_DC_Pessi_Finetune" = "#4363D8", 
        "CV_Search_Finetune" = "#F58231",
        "Supervised_All" = "#911EB4",
        "Only_Unbiased" = "#42D4F4"
      ),
      guide = guide_legend(
        nrow = 2,
        title.position = "top"
      )
    ) +
    theme_bw() +
    labs(
      title = paste("Performance Distribution by Method and Data Percentage:", name),
      x = "Percentage",
      y = "Metric Value",
      color = "Method"
    ) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 11, face = "bold"),
      axis.text.y = element_text(size = 10, face = "bold"),
      axis.title = element_text(size = 12, face = "bold"),
      legend.position = "bottom",
      legend.text = element_text(size = 9,face = "bold"),
      legend.title = element_text(size = 10, face = "bold"),
      strip.text = element_text(size = 11, face = "bold"),
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      panel.spacing = unit(0.8, "lines")
    )
  
  # 4. 保存高分辨率图像
  ggsave(
    paste0(name, "_beeswarm_plot.png"),
    p_facet,
    dpi = 600,
    width = 16,
    height = 9,
    units = "in"
  )
  
  # 返回转换后的长格式数据
  return(dt_long)
}

# 
# analyze_and_plot <- function(dt_select, name) {
#   
#   # dt_select = dt_select
#   # name = "DeepFM_Larger"
#   # 
#   
#   # 参数校验
#   if (!all(c("auc", "test_loss", "method", "percentage") %in% colnames(dt_select))) {
#     stop("数据集必须包含 auc, test_loss, method, percentage 列")
#   }
#   
#   # 1. 数据预处理
#   method_levels <- c("Pretrained_Tuning", "Loss_DC_Nonpessi_Tuning",
#                      "Loss_DC_Pessi_Tuning", "CV_Search_Tuning",
#                      "Supervised_All", "Only_Tuning")
#   #percentage_levels <- c("0%", "5%", "10%", "15%", "20%", "25%")
#   percentage_levels<-c("50%","70%","100%")
#   
#   dt_select <- dt_select %>%
#     mutate(
#       method = factor(method, levels = method_levels),
#       percentage = factor(percentage, levels = percentage_levels)
#     )
#   
#   # 2. 计算分组均值与标准差
#   # 2. 计算分组统计量（修复列名格式）
#   dt_avg <- dt_select %>%
#     group_by(method, percentage) %>%
#     summarise(
#       mean_auc = mean(auc, na.rm = TRUE),
#       sd_auc = sd(auc, na.rm = TRUE),
#       mean_testloss = mean(test_loss, na.rm = TRUE),  # 修改列名
#       sd_testloss = sd(test_loss, na.rm = TRUE),      # 使用testloss替代test_loss
#       .groups = "drop"
#     ) %>%
#     pivot_longer(
#       cols = -c(method, percentage),
#       names_to = c("stat_type", "metric"),
#       names_sep = "_",    # 现在所有列名都只有单个下划线
#       values_to = "value"
#     ) %>%
#     pivot_wider(
#       names_from = stat_type,
#       values_from = value
#     ) %>%
#     mutate(
#       metric = case_when(
#         metric == "auc" ~ "AUC",
#         metric == "testloss" ~ "Test Loss",  # 对应新列名
#         TRUE ~ metric
#       )
#     )
#   
#   
#   # 3. 保存结果
#   write.csv(dt_avg, paste0(name, "_method_percentage_averages.csv"), row.names = FALSE)
#   
#   # 4. 创建分组条形图
#   p_facet <- ggplot(dt_avg, aes(x = percentage, y = mean, fill = method)) +
#     geom_col(
#       position = position_dodge(0.9),  # 分组间距
#       width = 0.8,                     # 条形宽度
#       color = "white",                 # 条形边框
#       alpha = 0.9
#     ) +
#     geom_errorbar(
#       aes(ymin = mean - sd, ymax = mean + sd),
#       position = position_dodge(0.9),   # 与条形对齐
#       width = 0.25,                     # 误差棒宽度
#       color = "gray30",                # 误差棒颜色
#       size = 0.6                       # 误差棒粗细
#     ) +
#     facet_wrap(
#       ~ metric,
#       scales = "free_y",
#       ncol = 2,
#       labeller = labeller(metric = label_value)
#     ) +
#     scale_fill_manual(
#       values = c(
#         "Pretrained_Tuning" = "#E6194B",
#         "Loss_DC_Nonpessi_Tuning" = "#3CB44B",
#         "Loss_DC_Pessi_Tuning" = "#4363D8",
#         "CV_Search_Tuning" = "#F58231",
#         "Supervised_All" = "#911EB4",
#         "Only_Tuning" = "#42D4F4"
#       ),
#       guide = guide_legend(
#         nrow = 2,
#         title.position = "top",
#         override.aes = list(alpha = 1)
#       )
#     ) +
#     theme_bw() +
#     labs(
#       title = paste("Model Performance by Method and Data Percentage:", name),
#       x = "Unbiased Data Percentage",
#       y = "Metric Value (Mean ± SD)",
#       fill = "Method"
#     ) +
#     theme(
#       axis.text.x = element_text(angle = 45, hjust = 1, size = 11, face = "bold"),
#       axis.text.y = element_text(size = 10, face = "bold"),
#       axis.title = element_text(size = 12, face = "bold"),
#       legend.position = "bottom",
#       legend.text = element_text(size = 9),
#       legend.title = element_text(size = 10, face = "bold"),
#       strip.text = element_text(size = 11, face = "bold"),
#       plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
#       panel.spacing = unit(0.8, "lines")
#     )
#   
#   # 5. 保存高分辨率图像
#   ggsave(
#     paste0(name, "_grouped_barchart.png"),
#     p_facet,
#     dpi = 600,
#     width = 16,
#     height = 9,
#     units = "in"
#   )
#   
#   # 返回计算结果
#   return(dt_avg)
# }


# 
# analyze_and_plot <- function(dt_select, name) {
#   # 参数校验
#   if (!all(c("auc", "test_loss", "method", "percentage") %in% colnames(dt_select))) {
#     stop("数据集必须包含 auc, test_loss, method, percentage 列")
#   }
#   
#   # 1. 数据预处理
#   method_levels <- c("Pretrained_Tuning", "Loss_DC_Nonpessi_Tuning", 
#                      "Loss_DC_Pessi_Tuning", "CV_Search_Tuning", 
#                      "Supervised_All", "Only_Tuning")
#   
#   percentage_levels <- c("0%", "5%", "10%", "15%", "20%", "25%")
#   
#   dt_select <- dt_select %>%
#     mutate(
#       method = factor(method, levels = method_levels),
#       percentage = factor(percentage, levels = percentage_levels)
#     )
#   
#   # 2. 计算分组平均值
#   dt_avg <- dt_select %>%
#     group_by(method, percentage) %>%
#     summarise(
#       mean_auc = round(mean(auc, na.rm = TRUE), 5),
#       mean_test_loss = round(mean(test_loss, na.rm = TRUE), 5),
#       .groups = "drop"
#     )
#   
#   # 3. 保存结果
#   write.csv(dt_avg, paste0(name, "_method_percentage_averages.csv"), row.names = FALSE)
#   
#   # 4. 绘图函数
#   create_plot <- function(data, y_var, title_suffix, y_label) {
#     ggplot(data, aes(x = percentage, y = .data[[y_var]], fill = method)) +
#       geom_boxplot(
#         position = position_dodge(0.8),
#         alpha = 0.8,
#         outlier.shape = NA,
#         size = 0.5
#       ) +
#       stat_summary(
#         fun = mean, 
#         geom = "point", 
#         shape = 15, 
#         size = 1,
#         color = "black",
#         position = position_dodge(0.8)
#       ) +
#       stat_summary(
#         fun.data = mean_cl_normal,
#         geom = "errorbar",
#         width = 0.2,
#         color = "black",
#         size = 0.4,
#         position = position_dodge(0.8)
#       ) +
#       theme_bw() +
#       labs(
#         title = paste(y_label, "by Method and Percentage:", name),
#         x = "Percentage",
#         y = y_label,
#         fill = "Method"
#       ) +
#       theme(
#         axis.text.x = element_text(angle = 45, hjust = 1, size = 15,face = "bold"),
#         axis.text.y = element_text(size = 15,face = "bold"),
#         legend.position = "bottom",
#         legend.text = element_text(size = 14,face = "bold"),
#         legend.title = element_text(size = 14,face = "bold"),
#         plot.title = element_text(hjust = 0.5, face = "bold", size = 15)
#       ) +
#       scale_fill_brewer(palette = "Set1")
#   }
#   
#   # 5. 创建并保存图形
#   p_auc <- create_plot(dt_select, "auc", name, "AUC")
#   ggsave(paste0(name, "_auc_boxplot.png"), p_auc, dpi = 300, width = 10, height = 6)
#   
#   p_loss <- create_plot(dt_select, "test_loss", name, "Test Loss")
#   ggsave(paste0(name, "_test_loss_boxplot.png"), p_loss, dpi = 300, width = 10, height = 6)
#   
#   # 返回计算结果
#   return(dt_avg)
# }


# analyze_and_plot <- function(dt_select, name) {
#   # 参数校验
#   if (!all(c("auc", "test_loss", "method", "percentage") %in% colnames(dt_select))) {
#     stop("数据集必须包含 auc, test_loss, method, percentage 列")
#   }
#   
#   # 1. 数据预处理
#   method_levels <- c("Pretrained_Tuning", "Loss_DC_Nonpessi_Tuning", 
#                      "Loss_DC_Pessi_Tuning", "CV_Search_Tuning", 
#                      "Supervised_All", "Only_Tuning")
#   
#   percentage_levels <- c("50%","70%","100%")
#   
#   dt_select <- dt_select %>%
#     mutate(
#       method = factor(method, levels = method_levels),
#       percentage = factor(percentage, levels = percentage_levels)
#     )
#   
#   # 2. 计算分组平均值
#   dt_avg <- dt_select %>%
#     group_by(method, percentage) %>%
#     summarise(
#       mean_auc = round(mean(auc, na.rm = TRUE), 5),
#       mean_test_loss = round(mean(test_loss, na.rm = TRUE), 5),
#       .groups = "drop"
#     )
#   
#   # 3. 保存结果
#   write.csv(dt_avg, paste0(name, "_method_percentage_averages.csv"), row.names = FALSE)
#   
#   # 4. 绘图函数
#   create_plot <- function(data, y_var, title_suffix, y_label) {
#     ggplot(data, aes(x = percentage, y = .data[[y_var]], fill = method)) +
#       geom_boxplot(
#         position = position_dodge(0.8),
#         alpha = 0.8,
#         outlier.shape = NA,
#         size = 0.5
#       ) +
#       stat_summary(
#         fun = mean, 
#         geom = "point", 
#         shape = 15, 
#         size = 1,
#         color = "black",
#         position = position_dodge(0.8)
#       ) +
#       stat_summary(
#         fun.data = mean_cl_normal,
#         geom = "errorbar",
#         width = 0.2,
#         color = "black",
#         size = 0.4,
#         position = position_dodge(0.8)
#       ) +
#       theme_bw() +
#       labs(
#         title = paste(y_label, "by Method and Percentage:", name),
#         x = "Percentage",
#         y = y_label,
#         fill = "Method"
#       ) +
#       theme(
#         axis.text.x = element_text(angle = 45, hjust = 1, size = 15,face = "bold"),
#         axis.text.y = element_text(size = 15,face = "bold"),
#         legend.position = "bottom",
#         legend.text = element_text(size = 8,face = "bold"),
#         legend.title = element_text(size = 10,face = "bold"),
#         plot.title = element_text(hjust = 0.5, face = "bold", size = 15)
#       ) +
#       scale_fill_brewer(palette = "Set1")
#   }
#   
#   # 5. 创建并保存图形
#   p_auc <- create_plot(dt_select, "auc", name, "AUC")
#   ggsave(paste0(name, "_auc_boxplot.png"), p_auc, dpi = 300, width = 10, height = 6)
#   
#   p_loss <- create_plot(dt_select, "test_loss", name, "Test Loss")
#   ggsave(paste0(name, "_test_loss_boxplot.png"), p_loss, dpi = 300, width = 10, height = 6)
#   
#   # 返回计算结果
#   return(dt_avg)
# }

# analyze_and_plot <- function(dt_select, name) {
#   # 参数校验
#   if (!all(c("auc", "test_loss", "method", "percentage") %in% colnames(dt_select))) {
#     stop("数据集必须包含 auc, test_loss, method, percentage 列")
#   }
#   
#   # 1. 数据预处理
#   method_levels <- c("Pretrained_Tuning", "Loss_DC_Nonpessi_Tuning", 
#                      "Loss_DC_Pessi_Tuning", "CV_Search_Tuning", 
#                      "Supervised_All", "Only_Tuning")
#   
#   percentage_levels <- c("50%","70%","100%")
#   
#   dt_select <- dt_select %>%
#     mutate(
#       method = factor(method, levels = method_levels),
#       percentage = factor(percentage, levels = percentage_levels)
#     )
#   
#   # 2. 计算分组平均值
#   dt_avg <- dt_select %>%
#     group_by(method, percentage) %>%
#     summarise(
#       mean_auc = round(mean(auc, na.rm = TRUE), 5),
#       mean_test_loss = round(mean(test_loss, na.rm = TRUE), 5),
#       .groups = "drop"
#     )
#   
#   # 3. 保存结果
#   write.csv(dt_avg, paste0(name, "_method_percentage_averages.csv"), row.names = FALSE)
#   
#   # 4. 数据重塑为长格式
#   dt_long <- dt_select %>%
#     pivot_longer(
#       cols = c(auc, test_loss),
#       names_to = "metric",
#       values_to = "value"
#     ) %>%
#     mutate(
#       metric = factor(
#         metric,
#         levels = c("auc", "test_loss"),
#         labels = c("AUC", "Test Loss")
#       )
#     )
#   
#   # 5. 创建分面图
#   p_facet <- ggplot(dt_long, aes(x = percentage, y = value, fill = method)) +
#     geom_boxplot(
#       position = position_dodge(0.8),
#       alpha = 0.8,
#       outlier.shape = NA,
#       size = 0.5
#     ) +
#     # 添加均值标记
#     stat_summary(
#       fun = mean, 
#       geom = "point", 
#       shape = 15, 
#       size = 1,
#       color = "black",
#       position = position_dodge(0.8)
#     ) +
#     # 分面设置
#     facet_wrap(
#       ~ metric, 
#       scales = "free_y",  # y轴独立刻度
#       ncol = 2,          # 单行排列
#       labeller = labeller(metric = label_value)
#     ) +
#     # 主题美化
#     theme_bw() +
#     labs(
#       title = paste("Model Performance by Method and Percentage:", name),
#       x = "Fine-tuning Data Percentage",
#       y = "Metric Value",
#       fill = "Method"
#     ) +
#     theme(
#       axis.text.x = element_text(angle = 45, hjust = 1, size = 12, face = "bold"),
#       axis.text.y = element_text(size = 12, face = "bold"),
#       axis.title = element_text(size = 14, face = "bold"),
#       legend.position = "bottom",
#       legend.text = element_text(size = 10, face = "bold"),
#       legend.title = element_text(size = 12, face = "bold"),
#       strip.text = element_text(size = 12, face = "bold"),  # 分面标题
#       plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
#       panel.spacing = unit(1, "lines")  # 分面间距
#     ) +
#     # 保持原有配色方案
#     scale_fill_brewer(palette = "Set1") 
#   
#   # 6. 保存图像
#   ggsave(
#     paste0(name, "_faceted_performance.png"), 
#     p_facet, 
#     dpi = 300, 
#     width = 14,   # 增加宽度适应分面
#     height = 7
#   )
#   
#   # 返回计算结果
#   return(dt_avg)
# }


#dt<-read.csv("df_all_summary_DCN.csv",sep=",")
#dt<-read.csv("df_all_summary_FinalMLP.csv",sep=",")
#dt<-read.csv("df_all_summary_WideDeep.csv",sep=",")
dt<-read.csv("df_all_summary_DeepFM_Larger.csv",sep=",")
colnames(dt)
dt_select<-dt[,1:4]

dt_select <- dt_select %>%
  mutate(method = recode(method,
                         "Pretrained_Tuning" = "Pretrained_Finetune",
                         "Loss_DC_Nonpessi_Tuning" = "Loss_DC_Nonpessi_Finetune",
                         "Loss_DC_Pessi_Tuning" = "Loss_DC_Pessi_Finetune", 
                         "CV_Search_Tuning" = "CV_Search_Finetune",
                         "Only_Tuning" = "Only_Unbiased",
                         "Supervised_All" = "Supervised_All"  # 保持不变
  ))

unique(dt_select$method)

# 使用示例
result <- analyze_and_plot(dt_select = dt_select, name = "DeepFM_Larger")



library(tidyverse)
library(ggplot2)
library(dplyr)
library(ggplot2)
dt1<-read.csv("df_all_summary_DeepFM_128.csv",sep=",")
dt2<-read.csv("df_all_summary_DeepFM_256.csv",sep=",")
dt3<-read.csv("df_all_summary_DeepFM_512.csv",sep=",")
dt4<-read.csv("df_all_summary_DeepFM_1024.csv",sep=",")
dt1$batch_size=128
dt2$batch_size=256
dt3$batch_size=512
dt4$batch_size=1024

dt_all<-rbind(dt1,dt2,dt3,dt4)

dt_all_sub1<-dt_all[,-c(5,6)]


dt_all_sub1 <- dt_all_sub1 %>%
  mutate(method = recode(method,
                         "Pretrained_Tuning" = "Pretrained_Finetune",
                         "Loss_DC_Nonpessi_Tuning" = "Loss_DC_Nonpessi_Finetune",
                         "Loss_DC_Pessi_Tuning" = "Loss_DC_Pessi_Finetune", 
                         "CV_Search_Tuning" = "CV_Search_Finetune",
                         "Only_Tuning" = "Only_Unbiased",
                         "Supervised_All" = "Supervised_All"  # 保持不变
  ))


unique(dt_all_sub1$method)

# 1. 数据预处理
method_levels <- c("Pretrained_Finetune", "Loss_DC_Nonpessi_Finetune",
                   "Loss_DC_Pessi_Finetune", "CV_Search_Finetune",
                   "Supervised_All", "Only_Unbiased")
percentage_levels <- c("0%", "5%", "10%", "15%", "20%", "25%")
batch_size_levels<-c("128","256","512","1024")


dt_select <- dt_all_sub1 %>%
  mutate(
    method = factor(method, levels = method_levels),
    percentage = factor(percentage, levels = percentage_levels),
    batch_size=factor(batch_size,levels=batch_size_levels)
  )



# 修改为蜂群图绘图函数
create_beeswarm_plot <- function(data, metric_name, y_label) {
  # 直接使用原始数据，无需聚合
  ggplot(data, aes(x = percentage, y = .data[[metric_name]], color = method)) +
    ggbeeswarm::geom_quasirandom(
      dodge.width = 0.8,      # 设置分组间距
      size = 2,               # 点大小
      alpha = 0.7,            # 透明度
      width = 0.2             # 点分布宽度
    ) +
    facet_wrap(
      ~ batch_size,
      nrow = 2,
      scales = "free_y",
      labeller = labeller(batch_size = function(x) paste("Batch Size:", x))
    ) +
    theme_bw(base_size = 14) +
    labs(
      title = paste(y_label, "Distribution by Method, Percentage, and Batch Size"),
      x = "Percentage",
      y = y_label,
      color = "Method"
    ) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 12, face = "bold"),
      axis.text.y = element_text(size = 12, face = "bold"),
      axis.title = element_text(size = 14, face = "bold"),
      strip.text = element_text(size = 12, face = "bold"),
      legend.position = "bottom",
      legend.text = element_text(size = 12, face = "bold"),
      legend.title = element_text(size = 12, face = "bold"),
      plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
      panel.spacing = unit(1.2, "lines")
    ) +
    scale_color_manual(
      values = c(
        "Pretrained_Finetune" = "#E6194B",
        "Loss_DC_Nonpessi_Finetune" = "#3CB44B",
        "Loss_DC_Pessi_Finetune" = "#4363D8",
        "CV_Search_Finetune" = "#F58231",
        "Supervised_All" = "#911EB4",
        "Only_Unbiased" = "#42D4F4"
      ),
      guide = guide_legend(
        nrow = 2,
        title.position = "top",
        override.aes = list(alpha = 1, size = 3)
      )
    ) +
    guides(color = guide_legend(nrow = 2))
}

# 生成并保存图形（需要先安装ggbeeswarm包）
# install.packages("ggbeeswarm")
library(ggbeeswarm)

p_auc_swarm <- create_beeswarm_plot(dt_select, "auc", "AUC")
ggsave(
  "auc_beeswarm_plot.png",
  p_auc_swarm,
  dpi = 400,
  width = 16,
  height = 10,
  bg = "white"
)

p_loss_swarm <- create_beeswarm_plot(dt_select, "test_loss", "Test Loss")
ggsave(
  "test_loss_beeswarm_plot.png",
  p_loss_swarm,
  dpi = 400,
  width = 16,
  height = 10,
  bg = "white"
)


# library(ggplot2)
# library(dplyr)
# 
# 
# # 修改后的绘图函数
# create_grouped_barchart <- function(data, metric_name, y_label) {
#   # 计算均值和标准差
#   plot_data <- data %>%
#     group_by(method, percentage, batch_size) %>%
#     summarise(
#       mean_val = mean(.data[[metric_name]], na.rm = TRUE),
#       sd_val = sd(.data[[metric_name]], na.rm = TRUE),
#       .groups = "drop"
#     )
#   
#   # 创建分组条形图
#   ggplot(plot_data, aes(x = percentage, y = mean_val, fill = method)) +
#     geom_col(
#       position = position_dodge(0.9),
#       width = 0.7,
#       color = "white",
#       alpha = 0.9
#     ) +
#     geom_errorbar(
#       aes(ymin = mean_val - sd_val, ymax = mean_val + sd_val),
#       position = position_dodge(0.9),
#       width = 0.25,
#       color = "gray30",
#       size = 0.6
#     ) +
#     facet_wrap(
#       ~ batch_size,
#       nrow = 2,
#       scales = "free_y",
#       labeller = labeller(batch_size = function(x) paste("Batch Size:", x))
#     ) +
#     theme_bw(base_size = 14) +
#     labs(
#       title = paste(y_label, "by Method, Percentage, and Batch Size"),
#       x = "Percentage",
#       y = y_label,
#       fill = "Method"
#     ) +
#     theme(
#       axis.text.x = element_text(angle = 45, hjust = 1, size = 12, face = "bold"),
#       axis.text.y = element_text(size = 12, face = "bold"),
#       axis.title = element_text(size = 14, face = "bold"),
#       strip.text = element_text(size = 12, face = "bold"),
#       legend.position = "bottom",
#       legend.text = element_text(size = 12, face = "bold"),
#       legend.title = element_text(size = 12, face = "bold"),
#       plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
#       panel.spacing = unit(1.2, "lines")
#     ) +
#     scale_fill_manual(
#       values = c(
#         "Pretrained_Tuning" = "#E6194B",
#         "Loss_DC_Nonpessi_Tuning" = "#3CB44B",
#         "Loss_DC_Pessi_Tuning" = "#4363D8",
#         "CV_Search_Tuning" = "#F58231",
#         "Supervised_All" = "#911EB4",
#         "Only_Tuning" = "#42D4F4"
#       ),
#       guide = guide_legend(
#         nrow = 2,
#         title.position = "top",
#         override.aes = list(alpha = 1)
#       )
#     )+
#     guides(fill = guide_legend(nrow = 2))
# }
# 
# # 生成并保存图形
# p_auc <- create_grouped_barchart(dt_select, "auc", "AUC")
# ggsave(
#   "auc_grouped_barchart.png",
#   p_auc,
#   dpi = 400,
#   width = 16,
#   height = 10,
#   bg = "white"
# )
# 
# p_loss <- create_grouped_barchart(dt_select, "test_loss", "Test Loss") 
# ggsave(
#   "test_loss_grouped_barchart.png",
#   p_loss,
#   dpi = 400,
#   width = 16,
#   height = 10,
#   bg = "white"
# )

# 
# # 1. 修改绘图函数以支持分面
# create_faceted_plot <- function(data, y_var, y_label) {
#   ggplot(data, aes(x = percentage, y = .data[[y_var]], fill = method)) +
#     geom_boxplot(
#       position = position_dodge(0.8),
#       alpha = 0.8,
#       outlier.shape = NA,
#       size = 0.5
#     ) +
#     stat_summary(
#       fun = mean, 
#       geom = "point", 
#       shape = 10, 
#       size = 1.2,
#       color = "black",
#       position = position_dodge(0.8)
#     ) +
#     stat_summary(
#       fun.data = mean_cl_normal,
#       geom = "errorbar",
#       width = 0.2,
#       color = "black",
#       size = 0.4,
#       position = position_dodge(0.8)
#     ) +
#     facet_wrap(
#       ~ batch_size, 
#       nrow = 2, 
#       scales = "free_y",  # y轴自由刻度
#       labeller = labeller(batch_size = function(x) paste("Batch Size:", x))  # 分面标签前缀
#     ) +
#     theme_bw(base_size = 14) +  # 全局字体基准大小
#     labs(
#       title = paste(y_label, "by Method, Percentage, and Batch Size"),
#       x = "Percentage",
#       y = y_label,
#       fill = "Method"
#     ) +
#     theme(
#       axis.text.x = element_text(angle = 45, hjust = 1, size = 12, face = "bold"),
#       axis.text.y = element_text(size = 12, face = "bold"),
#       axis.title = element_text(size = 14, face = "bold"),
#       strip.text = element_text(size = 12, face = "bold"),  # 分面标签字体
#       legend.position = "bottom",
#       legend.text = element_text(size = 12, face = "bold"),
#       legend.title = element_text(size = 12, face = "bold"),
#       plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
#       panel.spacing = unit(1.2, "lines")  # 分面间距
#     ) +
#     scale_fill_brewer(
#       palette = "Set1",
#       labels = function(x) stringr::str_wrap(x, width = 15)  # 图例换行
#     ) +
#     guides(fill = guide_legend(nrow = 2, byrow = TRUE))  # 图例分两行
# }
# 
# # 2. 生成并保存图形
# # AUC 分面图
# p_auc <- create_faceted_plot(dt_select, "auc", "AUC")
# ggsave(
#   "auc_faceted_boxplot.png", 
#   p_auc, 
#   dpi = 400, 
#   width = 16, 
#   height = 10,
#   bg = "white"
# )
# 
# # Test Loss 分面图
# p_loss <- create_faceted_plot(dt_select, "test_loss", "Test Loss")
# ggsave(
#   "test_loss_faceted_boxplot.png", 
#   p_loss, 
#   dpi = 400, 
#   width = 16, 
#   height = 10,
#   bg = "white"
# )



library(tidyverse)

dt1<-read.csv("df_all_summary_DeepFM_128.csv",sep=",")
dt2<-read.csv("df_all_summary_DeepFM_256.csv",sep=",")
dt3<-read.csv("df_all_summary_DeepFM_512.csv",sep=",")
dt4<-read.csv("df_all_summary_DeepFM_1024.csv",sep=",")
dt1$batch_size=128
dt2$batch_size=256
dt3$batch_size=512
dt4$batch_size=1024

dt_all<-rbind(dt1,dt2,dt3,dt4)


dt_all <- dt_all %>%
  mutate(method = recode(method,
                         "Pretrained_Tuning" = "Pretrained_Finetune",
                         "Loss_DC_Nonpessi_Tuning" = "Loss_DC_Nonpessi_Finetune",
                         "Loss_DC_Pessi_Tuning" = "Loss_DC_Pessi_Finetune", 
                         "CV_Search_Tuning" = "CV_Search_Finetune",
                         "Only_Tuning" = "Only_Unbiased",
                         "Supervised_All" = "Supervised_All"  # 保持不变
  ))



dt_all_sub1<-dt_all[,-c(1,2)]

dt_all_sub2<-subset(dt_all_sub1,dt_all_sub1$method %in% c("Loss_DC_Nonpessi_Finetune",
                                                          "Loss_DC_Pessi_Finetune",
                                                          "CV_Search_Finetune"  ))

loss_dc_weight_data<-subset(dt_all_sub2,dt_all_sub2$method!="CV_Search_Finetune")[,-4]
CV_weight_data<-subset(dt_all_sub2,dt_all_sub2$method=="CV_Search_Finetune")[,-3]
loss_dc_weight_data<-as_tibble(loss_dc_weight_data)
CV_weight_data<-as_tibble(CV_weight_data)


method_levels <- c("Pretrained_Finetune", "Loss_DC_Nonpessi_Finetune", 
                   "Loss_DC_Pessi_Finetune", "CV_Search_Finetune", 
                   "Supervised_All", "Only_Finetune")
percentage_levels <- c("0%", "5%", "10%", "15%", "20%", "25%")
batch_size_levels<-c("128","256","512","1024")



dt_select1 <- CV_weight_data%>%
  mutate(
    method = factor(method, levels = method_levels),
    percentage = factor(percentage, levels = percentage_levels),
    batch_size=factor(batch_size,levels=batch_size_levels)
  )






dt_select1 %>% 
  distinct(method, percentage, batch_size) %>% 
  nrow()




names(dt_select1)

# 执行分组聚合
dt_avg_final <- dt_select1 %>%
  group_by(method, percentage, batch_size) %>%
  summarise(
    mean_weights_est = round(mean(weights_search, na.rm = TRUE), 5),
    .groups = "drop"
  ) 


# 2. 轨迹图绘制函数（带分面）
create_trajectory_plot <- function(data, y_var, y_label) {
  ggplot(data, aes(x = percentage, y = .data[[y_var]], 
                   group = method, color = method)) +
    # 轨迹线（带抖动避免重叠）
    geom_line(
      position = position_dodge(width = 0.3),  # 横向抖动
      linewidth = 1.2,
      alpha = 0.8
    ) +
    # 数据点
    geom_point(
      position = position_dodge(width = 0.3),  # 与线对齐
      size = 3,
      shape = 18  # 菱形标记
    ) +
    # 分面设置
    facet_wrap(
      ~ batch_size, 
      nrow = 2, 
      scales = "free_y",  # y轴自由刻度
      labeller = labeller(batch_size = function(x) paste("Batch Size:", x))  # 分面标签前缀
    ) +
    # 主题样式
    theme_bw(base_size = 14) +
    labs(
      title = paste(y_label, "Trajectory by Method and Batch Size"),
      x = "Percentage",
      y = y_label,
      color = "Method"
    ) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 12, face = "bold"),
      axis.text.y = element_text(size = 12, face = "bold"),
      axis.title = element_text(size = 14, face = "bold"),
      strip.text = element_text(size = 12, face = "bold", margin = margin(b = 10)),  # 分面标题加粗
      strip.background = element_rect(fill = "gray90"),  # 分面背景色
      legend.position = "bottom",
     # legend.position = "none",  # 关键修改：完全隐藏图例
      legend.text = element_text(size = 12, face = "bold"),
      legend.title = element_text(size = 12, face = "bold"),
      plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
      panel.spacing = unit(1.5, "lines")  # 增大分面间距
    ) +
    # 颜色方案
    scale_color_manual(
      values = c(
        "Pretrained_Finetune" = "#E6194B",
        "Loss_DC_Nonpessi_Finetune" = "#3CB44B",
        "Loss_DC_Pessi_Finetune" = "#4363D8",
        "CV_Search_Finetune" = "#F58231",
        "Supervised_All" = "#911EB4",
        "Only_Unbiased" = "#42D4F4"
      ),
      guide = guide_legend(
        nrow = 2,
        title.position = "top",
        override.aes = list(alpha = 1)
      )
    ) +
    # 优化坐标轴
    scale_y_continuous(n.breaks = 6)  # 保证每个分面有足够刻度
}

# 3. 使用示例
p_weights <- create_trajectory_plot(dt_avg_final, "mean_weights_est", "CV Weight")

# 4. 保存高清图（建议放大尺寸）
ggsave("CV_weights_trajectory.png", p_weights, 
       dpi = 300, width = 14, height = 10, units = "in")




##-----other modeling----


library(tidyverse)

#dt<-read.csv("df_all_summary_DCN.csv",sep=",")

dt<-read.csv("df_all_summary_FinalMLP.csv",sep=",")

#t<-read.csv("df_all_summary_WideDeep.csv",sep=",")


dt_all<-dt



dt_all <- dt_all %>%
  mutate(method = recode(method,
                         "Pretrained_Tuning" = "Pretrained_Finetune",
                         "Loss_DC_Nonpessi_Tuning" = "Loss_DC_Nonpessi_Finetune",
                         "Loss_DC_Pessi_Tuning" = "Loss_DC_Pessi_Finetune", 
                         "CV_Search_Tuning" = "CV_Search_Finetune",
                         "Only_Tuning" = "Only_Unbiased",
                         "Supervised_All" = "Supervised_All"  # 保持不变
  ))


dt_all_sub1<-dt_all[,-c(1,2)]

dt_all_sub2<-subset(dt_all_sub1,dt_all_sub1$method %in% c("Loss_DC_Nonpessi_Finetune",
                                                          "Loss_DC_Pessi_Finetune",
                                                          "CV_Search_Finetune"  ))

loss_dc_weight_data<-subset(dt_all_sub2,dt_all_sub2$method!="CV_Search_Finetune")[,-4]
CV_weight_data<-subset(dt_all_sub2,dt_all_sub2$method=="CV_Search_Finetune")[,-3]
loss_dc_weight_data<-as_tibble(loss_dc_weight_data)
CV_weight_data<-as_tibble(CV_weight_data)


method_levels <- c("Pretrained_Finetune", "Loss_DC_Nonpessi_Finetune", 
                   "Loss_DC_Pessi_Finetune", "CV_Search_Finetune", 
                   "Supervised_All", "Only_Unbiased")
percentage_levels <- c("0%", "5%", "10%", "15%", "20%", "25%")




dt_select1 <- loss_dc_weight_data %>%
  mutate(
    method = factor(method, levels = method_levels),
    percentage = factor(percentage, levels = percentage_levels)
  )



dt_select1 %>% 
  distinct(method, percentage) %>% 
  nrow()




names(dt_select1)

# 执行分组聚合
dt_avg_final <- dt_select1 %>%
  group_by(method, percentage) %>%
  summarise(
    mean_weight_est = round(mean(weights_est, na.rm = TRUE), 5),
    .groups = "drop"
  ) 


# 2. 轨迹图绘制函数（带分面）
create_trajectory_plot <- function(data, y_var, y_label) {
  ggplot(data, aes(x = percentage, y = .data[[y_var]], 
                   group = method, color = method)) +
    # 轨迹线（带抖动避免重叠）
    geom_line(
      position = position_dodge(width = 0.3),  # 横向抖动
      linewidth = 1.2,
      alpha = 0.8
    ) +
    # 数据点
    geom_point(
      position = position_dodge(width = 0.3),  # 与线对齐
      size = 3,
      shape = 18  # 菱形标记
    ) +
    # 主题样式
    theme_bw(base_size = 14) +
    labs(
      title = paste(y_label, "Trajectory by Method and Batch Size"),
      x = "Percentage",
      y = y_label,
      color = "Method"
    ) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 12, face = "bold"),
      axis.text.y = element_text(size = 12, face = "bold"),
      axis.title = element_text(size = 14, face = "bold"),
      strip.text = element_text(size = 12, face = "bold", margin = margin(b = 10)),  # 分面标题加粗
      strip.background = element_rect(fill = "gray90"),  # 分面背景色
      legend.position = "bottom",
      #legend.position = "none",  # 关键修改：完全隐藏图例
      legend.text = element_text(size = 12, face = "bold"),
      legend.title = element_text(size = 12, face = "bold"),
      plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
      panel.spacing = unit(1.5, "lines")  # 增大分面间距
    ) +
    # 颜色方案
    scale_color_manual(
      values = c(
        "Pretrained_Finetune" = "#E6194B",
        "Loss_DC_Nonpessi_Finetune" = "#3CB44B",
        "Loss_DC_Pessi_Finetune" = "#4363D8",
        "CV_Search_Finetune" = "#F58231",
        "Supervised_All" = "#911EB4",
        "Only_Unbiased" = "#42D4F4"
      ),
      guide = guide_legend(
        nrow = 2,
        title.position = "top",
        override.aes = list(alpha = 1)
      )
    ) +
    # 优化坐标轴
    scale_y_continuous(n.breaks = 6)  # 保证每个分面有足够刻度
}

# 3. 使用示例
p_weights <- create_trajectory_plot(dt_avg_final, "mean_weight_est", "FinalMLP: Mean Weight Estimate")

# 4. 保存高清图（建议放大尺寸）
ggsave("Loss_weights_trajectory_FinalMLP.png", p_weights, 
       dpi = 300, width = 14, height = 10, units = "in")






library(tidyverse)
library(ggplot2)

library(dplyr)
library(ggplot2)

# analyze_and_plot <- function(dt_select, name) {
#   
#   # dt_select = dt_select
#   # name = "DeepFM_Withoutfinetuning"
#   # # 
#   
#   # 参数校验
#   if (!all(c("auc", "test_loss", "method", "percentage") %in% colnames(dt_select))) {
#     stop("数据集必须包含 auc, test_loss, method, percentage 列")
#   }
#   
#   # 1. 数据预处理
#   method_levels <- c("Pretrained", "Loss_DC_Nonpessi",
#                      "Loss_DC_Pessi", "CV_Search",
#                     "Only_Tuning")
#   percentage_levels <- c("0%", "5%", "10%", "15%", "20%", "25%","30%","35%","40%")
# 
#   
#   dt_select <- dt_select %>%
#     mutate(
#       method = factor(method, levels = method_levels),
#       percentage = factor(percentage, levels = percentage_levels)
#     )
#   
#   # 2. 计算分组均值与标准差
#   # 2. 计算分组统计量（修复列名格式）
#   dt_avg <- dt_select %>%
#     group_by(method, percentage) %>%
#     summarise(
#       mean_auc = mean(auc, na.rm = TRUE),
#       sd_auc = sd(auc, na.rm = TRUE),
#       mean_testloss = mean(test_loss, na.rm = TRUE),  # 修改列名
#       sd_testloss = sd(test_loss, na.rm = TRUE),      # 使用testloss替代test_loss
#       .groups = "drop"
#     ) %>%
#     pivot_longer(
#       cols = -c(method, percentage),
#       names_to = c("stat_type", "metric"),
#       names_sep = "_",    # 现在所有列名都只有单个下划线
#       values_to = "value"
#     ) %>%
#     pivot_wider(
#       names_from = stat_type,
#       values_from = value
#     ) %>%
#     mutate(
#       metric = case_when(
#         metric == "auc" ~ "AUC",
#         metric == "testloss" ~ "Test Loss",  # 对应新列名
#         TRUE ~ metric
#       )
#     )
#   
#   
#   # 3. 保存结果
#   write.csv(dt_avg, paste0(name, "_method_percentage_averages.csv"), row.names = FALSE)
#   
#   # 4. 创建分组条形图
#   p_facet <- ggplot(dt_avg, aes(x = percentage, y = mean, fill = method)) +
#     geom_col(
#       position = position_dodge(0.9),  # 分组间距
#       width = 0.8,                     # 条形宽度
#       color = "white",                 # 条形边框
#       alpha = 0.9
#     ) +
#     geom_errorbar(
#       aes(ymin = mean - sd, ymax = mean + sd),
#       position = position_dodge(0.9),   # 与条形对齐
#       width = 0.25,                     # 误差棒宽度
#       color = "gray30",                # 误差棒颜色
#       size = 0.6                       # 误差棒粗细
#     ) +
#     facet_wrap(
#       ~ metric,
#       scales = "free_y",
#       ncol = 2,
#       labeller = labeller(metric = label_value)
#     ) +
#     scale_fill_manual(
#       values = c(
#         "Pretrained" = "#E6194B",
#         "Loss_DC_Nonpessi" = "#3CB44B",
#         "Loss_DC_Pessi" = "#4363D8",
#         "CV_Search" = "#F58231",
#         "Only_Tuning" = "#42D4F4"
#       ),
#       guide = guide_legend(
#         nrow = 2,
#         title.position = "top",
#         override.aes = list(alpha = 1)
#       )
#     ) +
#     theme_bw() +
#     labs(
#       title = paste("Model Performance by Method and Data Percentage:", name),
#       x = "Unbiased Data Percentage",
#       y = "Metric Value (Mean ± SD)",
#       fill = "Method"
#     ) +
#     theme(
#       axis.text.x = element_text(angle = 45, hjust = 1, size = 11, face = "bold"),
#       axis.text.y = element_text(size = 10, face = "bold"),
#       axis.title = element_text(size = 12, face = "bold"),
#       legend.position = "bottom",
#       legend.text = element_text(size = 9),
#       legend.title = element_text(size = 10, face = "bold"),
#       strip.text = element_text(size = 11, face = "bold"),
#       plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
#       panel.spacing = unit(0.8, "lines")
#     )
#   
#   # 5. 保存高分辨率图像
#   ggsave(
#     paste0(name, "_grouped_barchart.png"),
#     p_facet,
#     dpi = 600,
#     width = 16,
#     height = 9,
#     units = "in"
#   )
#   
#   # 返回计算结果
#   return(dt_avg)
# }


analyze_and_plot <- function(dt_select, name) {
  
  # 参数校验
  if (!all(c("auc", "test_loss", "method", "percentage") %in% colnames(dt_select))) {
    stop("数据集必须包含 auc, test_loss, method, percentage 列")
  }
  
  # 1. 数据预处理（保持原始数据格式）
  method_levels <- c("Pretrained", "Loss_DC_Nonpessi",
                       "Loss_DC_Pessi", "CV_Search",
                      "Only_Unbiased")
  
  percentage_levels <- c("0%", "5%", "10%", "15%", "20%", "25%","30%","35%","40%")

  dt_long <- dt_select %>%
    mutate(
      method = factor(method, levels = method_levels),
      percentage = factor(percentage, levels = percentage_levels)
    ) %>%
    pivot_longer(
      cols = c(auc, test_loss),
      names_to = "metric",
      values_to = "value"
    ) %>%
    mutate(
      metric = case_when(
        metric == "auc" ~ "AUC",
        metric == "test_loss" ~ "Test Loss",
        TRUE ~ metric
      )
    )
  
  # 2. 保存长格式数据
  write.csv(dt_long, paste0(name, "_long_format_data.csv"), row.names = FALSE)
  
  # 3. 创建蜂群图
  p_facet <- ggplot(dt_long, aes(x = percentage, y = value, color = method)) +
    # 蜂群图核心层
    ggbeeswarm::geom_beeswarm(
      dodge.width = 0.9,    # 分组间距
      size = 2.5,           # 点大小
      alpha = 0.8,          # 透明度
      method = "swarm",     # 排列方式
      cex = 0.8             # 点间距控制
    ) +
    # 分面显示指标
    facet_wrap(
      ~ metric,
      scales = "free_y",
      ncol = 2,
      labeller = labeller(metric = label_value)
    ) +
    # 颜色方案
    scale_color_manual(
      values =c(
                "Pretrained" = "#E6194B",
                "Loss_DC_Nonpessi" = "#3CB44B",
                "Loss_DC_Pessi" = "#4363D8",
                "CV_Search" = "#F58231",
                "Only_Unbiased" = "#42D4F4"
              ),
      guide = guide_legend(
        nrow = 2,
        title.position = "top"
      )
    ) +
    theme_bw() +
    labs(
      title = paste("Performance Distribution by Method and Data Percentage:", name),
      x = "Percentage",
      y = "Metric Value",
      color = "Method"
    ) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 11, face = "bold"),
      axis.text.y = element_text(size = 10, face = "bold"),
      axis.title = element_text(size = 12, face = "bold"),
      legend.position = "bottom",
      legend.text = element_text(size = 9,face = "bold"),
      legend.title = element_text(size = 10, face = "bold"),
      strip.text = element_text(size = 11, face = "bold"),
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      panel.spacing = unit(0.8, "lines")
    )
  
  # 4. 保存高分辨率图像
  ggsave(
    paste0(name, "_beeswarm_plot.png"),
    p_facet,
    dpi = 600,
    width = 16,
    height = 9,
    units = "in"
  )
  
  # 返回转换后的长格式数据
  return(dt_long)
}


dt<-read.csv("df_all_summary_DeepFM_Withoutfinetuning.csv",sep=",")
colnames(dt)
dt_select<-dt[,1:4]



dt_select <- dt_select %>%
  mutate(method = recode(method,
                         "Only_Tuning" = "Only_Unbiased"
  ))



# 使用示例
result <- analyze_and_plot(dt_select = dt_select, name = "DeepFM_Withoutfinetuning")




library(tidyverse)
library(ggplot2)

dt<-read.csv("df_all_summary_DeepFM_Withoutfinetuning.csv",sep=",")[,-c(1,2)]

dt_all_sub2<-subset(dt,dt$method %in% c("Loss_DC_Nonpessi",
                                                          "Loss_DC_Pessi",
                                                          "CV_Search"  ))

loss_dc_weight_data<-subset(dt_all_sub2,dt_all_sub2$method!="CV_Search")[,-4]
CV_weight_data<-subset(dt_all_sub2,dt_all_sub2$method=="CV_Search")[,-3]
loss_dc_weight_data<-as_tibble(loss_dc_weight_data)
CV_weight_data<-as_tibble(CV_weight_data)


method_levels <- c("Loss_DC_Nonpessi", 
                   "Loss_DC_Pessi", "CV_Search")
percentage_levels <- c("0%", "5%", "10%", "15%", "20%", "25%","30%","35%","40%")




dt_select1 <- loss_dc_weight_data %>%
  mutate(
    method = factor(method, levels = method_levels),
    percentage = factor(percentage, levels = percentage_levels)
  )



dt_select1 %>% 
  distinct(method, percentage) %>% 
  nrow()




names(dt_select1)

# 执行分组聚合
dt_avg_final <- dt_select1 %>%
  group_by(method, percentage) %>%
  summarise(
    mean_weight = round(mean(weights_est, na.rm = TRUE), 5),
    .groups = "drop"
  ) 


# 2. 轨迹图绘制函数（带分面）
create_trajectory_plot <- function(data, y_var, y_label) {
  ggplot(data, aes(x = percentage, y = .data[[y_var]], 
                   group = method, color = method)) +
    # 轨迹线（带抖动避免重叠）
    geom_line(
      position = position_dodge(width = 0.3),  # 横向抖动
      linewidth = 1.2,
      alpha = 0.8
    ) +
    # 数据点
    geom_point(
      position = position_dodge(width = 0.3),  # 与线对齐
      size = 3,
      shape = 18  # 菱形标记
    ) +
    # 主题样式
    theme_bw(base_size = 14) +
    labs(
      title = paste(y_label, "Trajectory by Method and Batch Size"),
      x = "Percentage",
      y = y_label,
      color = "Method"
    ) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 12, face = "bold"),
      axis.text.y = element_text(size = 12, face = "bold"),
      axis.title = element_text(size = 14, face = "bold"),
      strip.text = element_text(size = 12, face = "bold", margin = margin(b = 10)),  # 分面标题加粗
      strip.background = element_rect(fill = "gray90"),  # 分面背景色
      legend.position = "bottom",
      #legend.position = "none",  # 关键修改：完全隐藏图例
      legend.text = element_text(size = 12, face = "bold"),
      legend.title = element_text(size = 12, face = "bold"),
      plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
      panel.spacing = unit(1.5, "lines")  # 增大分面间距
    ) +
    # 颜色方案
    scale_color_manual(
      values = c(
        "Pretrained" = "#E6194B",
        "Loss_DC_Nonpessi" = "#3CB44B",
        "Loss_DC_Pessi" = "#4363D8",
        "CV_Search" = "#F58231",
        "Supervised_All" = "#911EB4",
        "Only_Tuning" = "#42D4F4"
      ),
      guide = guide_legend(
        nrow = 2,
        title.position = "top",
        override.aes = list(alpha = 1)
      )
    ) +
    # 优化坐标轴
    scale_y_continuous(n.breaks = 6)  # 保证每个分面有足够刻度
}

# 3. 使用示例
p_weights <- create_trajectory_plot(dt_avg_final, "mean_weight", "DeepFM_Withoutfinetuning: Mean Weight Estimate")

# 4. 保存高清图（建议放大尺寸）
ggsave("Loss_weights_trajectory_DeepFM_Withoutfinetuning.png", p_weights, 
       dpi = 300, width = 14, height = 10, units = "in")



