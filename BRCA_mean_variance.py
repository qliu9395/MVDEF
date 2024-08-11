import numpy as np

print("======BRCA_mean_variance:=======\n")
# 定义数据
acc_values = [
    0.859, 0.878, 0.867, 0.867, 0.871, 0.867, 0.844, 0.867, 0.863, 0.871
]

f1_weighted_values = [
    0.855, 0.878, 0.871, 0.870, 0.873, 0.866, 0.842, 0.866, 0.864, 0.871
]

f1_macro_values = [
    0.822, 0.850, 0.842, 0.837, 0.833, 0.838, 0.815, 0.839, 0.824, 0.828
]


# 计算平均值
acc_mean = np.mean(acc_values)
# 计算方差
acc_variance = np.var(acc_values)

print("ACC平均值: ",acc_mean)
print("ACC方差: ",acc_variance)
print()

# 计算平均值
f1_weighted_mean = np.mean(f1_weighted_values)
# 计算方差
f1_weighted_variance = np.var(f1_weighted_values)

print("F1_W平均值: ",f1_weighted_mean)
print("F1_W方差: ",f1_weighted_variance)
print()

# 计算平均值
f1_macro_mean = np.mean(f1_macro_values)
# 计算方差
f1_macro_variance = np.var(f1_macro_values)

print("F1_M平均值: ",f1_macro_mean)
print("F1_M方差: ",f1_macro_variance)

'''
Table 4 Classification results on BRCA dataset.

Method                ACC                  F1_weighted         F1_macro
--------------------------------------------------------------------
KNN                   0.742 ± 0.024        0.730 ± 0.023       0.682 ± 0.025
SVM                   0.729 ± 0.018        0.702 ± 0.015       0.640 ± 0.017
Lasso                 0.732 ± 0.012        0.698 ± 0.015       0.642 ± 0.026
RF                    0.754 ± 0.009        0.733 ± 0.010       0.649 ± 0.013
XGBoost               0.781 ± 0.008        0.764 ± 0.010       0.701 ± 0.017
NN                    0.754 ± 0.028        0.740 ± 0.034       0.668 ± 0.047
GRridge               0.745 ± 0.016        0.726 ± 0.019       0.656 ± 0.025
block PLSDA           0.642 ± 0.009        0.534 ± 0.014       0.369 ± 0.017
block sPLSDA          0.639 ± 0.008        0.522 ± 0.016       0.351 ± 0.022
NN_NN                 0.796 ± 0.012        0.784 ± 0.014       0.723 ± 0.018
NN_VCDN               0.792 ± 0.010        0.781 ± 0.006       0.721 ± 0.018
MOGONET_NN            0.805 ± 0.017        0.782 ± 0.030       0.737 ± 0.038
MOGONET               0.829 ± 0.018        0.825 ± 0.016       0.774 ± 0.017
TMC_MOGONET (Ours)    0.865 ± 0.009        0.866 ± 0.010       0.833 ± 0.010
'''