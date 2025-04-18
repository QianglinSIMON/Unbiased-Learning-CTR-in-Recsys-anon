# Unbiased-Learning-CTR-in-Recsys

CODE REPOSITORY DOCUMENTATION
=============================

1. Data Preprocessing
---------------------
Directory: kuai_rand_Datasets_preprocess/
- Function: Processes raw Kuai-Rand dataset
- Output Files:
  * Datasets/random_all_data.csv (random interaction data)
  * Datasets/normal_all_data.csv (normal interaction data)

2. DeepFM Model Implementation
------------------------------
2.1 Core Modules
- deepfm.py: Implements DeepFM model architecture
- data_preprocess_split.py:
  * Splits biased/unbiased datasets
  * Auxiliary functions:
    - get_metrics: Evaluation metrics calculation
    - custom_loss_function: Custom loss implementation
    - extract_time_features: Temporal feature extraction
    - generate_data: Data generation utility
    - Early stopping criteria

2.2 Pretraining
- pre_train_model.py: Supervised pretraining using full biased dataset

2.3 Finetuning Methods
- pre_trained_model_tuning.py: Pretrain-Finetune approach
- only_tuning.py: Supervised learning baseline (unbiased data only)

2.4 Ablation Studies
- loss_DC_Nonpessi_tuning.py: Loss Debiasing (Non-Pessimistic) + Finetuning
- loss_DC_Pessi_tuning.py: Loss Debiasing (Pessimistic) + Finetuning
- loss_DC_Nonpessi_pure.py: Standalone Loss_DC_Nonpessi method
- loss_DC_Pessi_pure.py: Standalone Loss_DC_Pessi method

2.5 Hyperparameter Optimization
- CV_search_finetuning.py: Cross-validated finetuning
- CV_search.py: Standard cross-validation
- grid_search_alpha.py: Fixed fusion weight search (with pretraining)
- grid_search_alpha_without_pretrain.py: Fixed fusion weight search (without pretraining)

2.6 Results Processing
- Summary_data_DeepFM_*.py: Experimental results reformatting scripts

3. Other Model Implementations：
All following directories contain analogous implementations to the DeepFM structure.
-----------------------------
- DCN/: Deep & Cross Network implementations
- WideDeep/: Wide & Deep model implementations
- FinalMLP/: Multilayer Perceptron implementations 







