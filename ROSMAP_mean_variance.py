import numpy as np

print("======ROSMAP_mean_variance:=======\n")
# 定义数据
acc_values = [
    0.830, 0.821, 0.858, 0.830, 0.802, 0.811, 0.840, 0.811, 0.840, 0.840
]

f1_values = [
    0.845, 0.816, 0.848, 0.824, 0.814, 0.811, 0.838, 0.831, 0.825, 0.838
]

auc_values = [
    0.913, 0.896, 0.901, 0.914, 0.911, 0.912, 0.905, 0.918, 0.898, 0.900
]


# 计算平均值
acc_mean = np.mean(acc_values)
# 计算方差
acc_variance = np.var(acc_values)

print("ACC平均值: ",acc_mean)
print("ACC方差: ",acc_variance)
print()

# 计算平均值
f1_mean = np.mean(f1_values)
# 计算方差
f1_variance = np.var(f1_values)

print("F1平均值: ",f1_mean)
print("F1方差: ",f1_variance)
print()

# 计算平均值
auc_mean = np.mean(auc_values)
# 计算方差
auc_variance = np.var(auc_values)

print("AUC平均值: ",auc_mean)
print("AUC方差: ",auc_variance)

'''
Table 2 Classification results on ROSMAP dataset.

Method                ACC                  F1                  AUC
--------------------------------------------------------------------
KNN                   0.657 ± 0.036        0.671 ± 0.044       0.709 ± 0.045
SVM                   0.770 ± 0.024        0.778 ± 0.016       0.770 ± 0.026
Lasso                 0.694 ± 0.037        0.730 ± 0.033       0.770 ± 0.035
RF                    0.726 ± 0.029        0.734 ± 0.021       0.811 ± 0.019
XGBoost               0.760 ± 0.046        0.772 ± 0.045       0.837 ± 0.030
NN                    0.755 ± 0.021        0.764 ± 0.021       0.827 ± 0.025
GRridge               0.760 ± 0.034        0.769 ± 0.029       0.841 ± 0.023
block PLSDA           0.742 ± 0.024        0.755 ± 0.023       0.830 ± 0.025
block sPLSDA          0.753 ± 0.033        0.764 ± 0.035       0.838 ± 0.021
NN_NN                 0.766 ± 0.023        0.777 ± 0.019       0.819 ± 0.017
NN_VCDN               0.775 ± 0.026        0.790 ± 0.018       0.843 ± 0.021
MOGONET_NN            0.804 ± 0.016        0.808 ± 0.010       0.858 ± 0.024
MOGONET               0.815 ± 0.023        0.821 ± 0.022       0.874 ± 0.012
TMC_MOGONET (Ours)    0.828 ± 0.017        0.829 ± 0.015       0.907 ± 0.000
'''