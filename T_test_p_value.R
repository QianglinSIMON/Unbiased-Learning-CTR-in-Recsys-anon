#dt<-read.csv("df_all_summary_DCN.csv",sep=",")
#dt<-read.csv("df_all_summary_FinalMLP.csv",sep=",")
dt<-read.csv("df_all_summary_WideDeep.csv",sep=",")
#dt<-read.csv("df_all_summary_DeepFM_Larger.csv",sep=",")
colnames(dt)
dt_select<-dt[,1:4]
unique(dt_select$method)

dt_select_sub<-subset(dt_select,dt_select$method %in% c("Pretrained_Tuning" ,
                                                        "Loss_DC_Nonpessi_Tuning",
                                                        "Loss_DC_Pessi_Tuning"  ))

val1<-subset(dt_select,dt_select$method=="Loss_DC_Nonpessi_Tuning"&dt_select$percentage=="25%")

val2<-subset(dt_select,dt_select$method=="Pretrained_Tuning" & dt_select$percentage=="25%")

val1$auc

val2$auc


library(dplyr)

# 预处理数据：过滤0%数据并转换百分比为因子
dt_filtered <- dt_select_sub %>%
  filter(percentage != "0%") %>%
  mutate(percentage = factor(percentage))

# 获取所有需要比较的百分比（排除0%）
percentages <- levels(dt_filtered$percentage)

# 定义Mann-Whitney检验函数
perform_tests <- function(perc, metric) {
  # 提取当前百分比下各方法数据
  pretrained <- dt_filtered %>% 
    filter(percentage == perc, method == "Pretrained_Tuning") %>% 
    pull(!!sym(metric))
  
  nonpessi <- dt_filtered %>% 
    filter(percentage == perc, method == "Loss_DC_Nonpessi_Tuning") %>% 
    pull(!!sym(metric))
  
  pessi <- dt_filtered %>% 
    filter(percentage == perc, method == "Loss_DC_Pessi_Tuning") %>% 
    pull(!!sym(metric))
  
  # 执行检验
  list(
    # Nonpessi vs Pretrained
    nonpessi_auc = if(metric == "auc") {
      wilcox.test(nonpessi, pretrained, alternative = "greater", 
                  exact = F, correct = F)$p.value
    } else {
      wilcox.test(nonpessi, pretrained, alternative = "two.sided", 
                  exact = F, correct = F)$p.value
    },
    
    # Pessi vs Pretrained
    pessi_auc = if(metric == "auc") {
      wilcox.test(pessi, pretrained, alternative = "greater", 
                  exact = F, correct = F)$p.value
    } else {
      wilcox.test(pessi, pretrained, alternative = "two.sided", 
                  exact = F, correct = F)$p.value
    }
  )
}

# 对每个百分比执行检验
result_list <- lapply(percentages, function(perc) {
  auc_res <- perform_tests(perc, "auc")
  loss_res <- perform_tests(perc, "test_loss")
  
  data.frame(
    percentage = perc,
    comparison = rep(c("Nonpessi_vs_Pretrained", "Pessi_vs_Pretrained"), 2),
    metric = rep(c("AUC", "Loss"), each = 2),
    p_value = c(auc_res$nonpessi_auc, auc_res$pessi_auc, 
                loss_res$nonpessi_auc, loss_res$pessi_auc)
  )
}) %>% bind_rows()

# 格式化输出结果
final_result <- result_list %>%
  mutate(
    significance = ifelse(p_value < 0.05, 
                          ifelse(p_value < 0.01, "**", "*"), 
                          "NS"),
    p_value = format.pval(p_value, digits = 4, eps = 0.0001)
  ) %>%
  arrange(percentage, metric, comparison)

# 显示结果
print(final_result)

write.csv(final_result,"WideDeep_compare_tests.csv")



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

dt_select_sub<-subset(dt_all_sub1,dt_all_sub1$method %in% c("Pretrained_Tuning" ,
                                                        "Loss_DC_Nonpessi_Tuning",
                                                        "Loss_DC_Pessi_Tuning"  ))


library(dplyr)
library(purrr)

# 预处理数据：按批次和百分比分组
dt_filtered <- dt_select_sub %>%
  filter(percentage != "0%") %>%
  mutate(
    percentage = factor(percentage),
    batch_size = factor(batch_size)
  ) 

# 获取所有批次和百分比的组合
group_combos <- dt_filtered %>%
  distinct(batch_size, percentage) %>%
  arrange(batch_size, percentage)

# 定义增强版检验函数
perform_group_tests <- function(bs, perc, metric) {
  # 提取当前批次和百分比下的数据
  pretrained <- dt_filtered %>% 
    filter(batch_size == bs, percentage == perc, 
           method == "Pretrained_Tuning") %>% 
    pull(!!sym(metric))
  
  nonpessi <- dt_filtered %>% 
    filter(batch_size == bs, percentage == perc,
           method == "Loss_DC_Nonpessi_Tuning") %>% 
    pull(!!sym(metric))
  
  pessi <- dt_filtered %>% 
    filter(batch_size == bs, percentage == perc,
           method == "Loss_DC_Pessi_Tuning") %>% 
    pull(!!sym(metric))
  
  # 执行检验并处理异常
  safe_wilcox <- function(...) {
    res <- suppressWarnings(
      tryCatch(wilcox.test(...),
               error = function(e) list(p.value = NA))
    )
    res$p.value
  }
  
  list(
    nonpessi_auc = if(metric == "auc") {
      safe_wilcox(nonpessi, pretrained, alternative = "greater", 
                  exact = FALSE, correct = FALSE)
    } else {
      safe_wilcox(nonpessi, pretrained, alternative = "two.sided",
                  exact = FALSE, correct = FALSE)
    },
    pessi_auc = if(metric == "auc") {
      safe_wilcox(pessi, pretrained, alternative = "greater", 
                  exact = FALSE, correct = FALSE)
    } else {
      safe_wilcox(pessi, pretrained, alternative = "two.sided",
                  exact = FALSE, correct = FALSE)
    }
  )
}

# 主执行流程
result_list <- map2_dfr(
  group_combos$batch_size, 
  group_combos$percentage,
  function(bs, perc) {
    auc_res <- perform_group_tests(bs, perc, "auc")
    loss_res <- perform_group_tests(bs, perc, "test_loss")
    
    data.frame(
      batch_size = bs,
      percentage = perc,
      comparison = rep(c("Nonpessi_vs_Pretrained", "Pessi_vs_Pretrained"), 2),
      metric = rep(c("AUC", "Loss"), each = 2),
      p_value = c(auc_res$nonpessi_auc, auc_res$pessi_auc, 
                  loss_res$nonpessi_auc, loss_res$pessi_auc)
    )
  }
)

# 结果后处理
final_result <- result_list %>%
  mutate(
    significance = case_when(
      p_value < 0.001 ~ "***",
      p_value < 0.01 ~ "**",
      p_value < 0.05 ~ "*",
      TRUE ~ "NS"
    ),
    p_value = ifelse(is.na(p_value), "NA", 
                     format.pval(p_value, digits = 4, eps = 0.0001))
  ) %>%
  arrange(batch_size, percentage, metric, comparison)

# 输出交互式表格 
DT::datatable(
  final_result,
  options = list(pageLength = 20, autoWidth = TRUE),
  rownames = FALSE
)
final_result
