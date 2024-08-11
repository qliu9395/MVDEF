import copy
import torch
from utils import load_model_dict, one_hot_tensor
from models import init_model_dict
from train_test import prepare_trte_data, gen_trte_adj_mat, test_epoch
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
cuda = True if torch.cuda.is_available() else False

def cal_feat_imp(data_folder, model_folder, view_list, num_class):
    """
    计算特征重要性

    参数:
    data_folder (str): 数据文件夹名称
    model_folder (str): 模型文件夹路径
    view_list (list): 视图列表
    num_class (int): 分类数量

    返回:
    list: 特征重要性列表
    """

    def get_adjustment_params(data_folder):
        """
        获取调整参数

        参数:
        data_folder (str): 数据文件夹名称

        返回:
        tuple: 包含调整参数和隐藏层维度列表的元组
        """
        if data_folder == 'ROSMAP':
            return 2, [200, 200, 100]
        elif data_folder == 'BRCA':
            return 10, [400, 400, 200]
        else:
            raise ValueError("Unknown data folder")

    def load_feature_names(data_folder, view_list):
        """
        加载特征名称

        参数:
        data_folder (str): 数据文件夹名称
        view_list (list): 视图列表

        返回:
        list: 特征名称列表
        """
        return [pd.read_csv(os.path.join(data_folder, f"{v}_featname.csv"), header=None).values.flatten() for v in view_list]

    def calculate_f1(labels, preds, num_class):
        """
        计算F1分数

        参数:
        labels (array): 真实标签
        preds (array): 预测标签
        num_class (int): 分类数量

        返回:
        float: F1分数
        """
        if num_class == 2:
            return f1_score(labels, preds)
        else:
            return f1_score(labels, preds, average='macro')

    # 获取调整参数和隐藏层维度列表
    adj_parameter, dim_he_list = get_adjustment_params(data_folder)

    # 准备数据
    data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_trte_data(data_folder, view_list)
    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)
    labels_te_tensor = torch.LongTensor(labels_trte[trte_idx["te"]])
    onehot_labels_te_tensor = one_hot_tensor(labels_te_tensor, num_class)

    # 加载特征名称
    featname_list = load_feature_names(data_folder, view_list)
    dim_list = [x.shape[1] for x in data_tr_list]

    # 初始化模型
    model_dict = init_model_dict(len(view_list), num_class, dim_list, dim_he_list)
    if torch.cuda.is_available():
        for m in model_dict.values():
            m.cuda()

    # 加载模型
    model_dict = load_model_dict(model_folder, model_dict)

    # 测试模型并获取预测结果
    _, _, _, all_preds = test_epoch(onehot_labels_te_tensor, data_trte_list, adj_te_list, trte_idx["te"], model_dict)

    # 计算初始的F1分数
    best_f1 = calculate_f1(labels_trte[trte_idx["te"]], all_preds.argmax(dim=1).cpu().detach().numpy(), num_class)

    # 初始化特征重要性列表
    feat_imp_list = []

    # 计算每个视图中的每个特征的重要性
    for i, feat_names in enumerate(featname_list):
        feat_imp = {"feat_name": feat_names, 'imp': np.zeros(dim_list[i])}

        for j in range(dim_list[i]):
            try:
                # 保存原始特征值
                feat_tr, feat_trte = data_tr_list[i][:, j].clone(), data_trte_list[i][:, j].clone()
                # 将特征值置零
                data_tr_list[i][:, j], data_trte_list[i][:, j] = 0, 0
                # 重新计算邻接矩阵
                adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)
                # 测试模型并获取新的预测结果
                _, _, _, all_preds = test_epoch(onehot_labels_te_tensor, data_trte_list, adj_te_list, trte_idx["te"], model_dict)
                # 计算新的F1分数
                f1_tmp = calculate_f1(labels_trte[trte_idx["te"]], all_preds.argmax(dim=1).cpu().detach().numpy(), num_class)
                # 计算特征重要性
                feat_imp['imp'][j] = (best_f1 - f1_tmp) * dim_list[i]
                # 恢复原始特征值
                data_tr_list[i][:, j], data_trte_list[i][:, j] = feat_tr, feat_trte
            except IndexError as e:
                print(f"IndexError: {e}, i: {i}, j: {j}, dim_list[i]: {dim_list[i]}")
                continue

        feat_imp_list.append(pd.DataFrame(feat_imp))

    return feat_imp_list

def summarize_imp_feat(featimp_list_list, topn=30):
    num_rep = len(featimp_list_list)
    num_view = len(featimp_list_list[0])
    df_tmp_list = []

    for v in range(num_view):
        df_tmp = copy.deepcopy(featimp_list_list[0][v])
        df_tmp['omics'] = np.ones(df_tmp.shape[0], dtype=int) * v
        df_tmp_list.append(df_tmp.copy(deep=True))

    df_featimp = pd.concat(df_tmp_list).copy(deep=True)

    for r in range(1, num_rep):
        for v in range(num_view):
            df_tmp = copy.deepcopy(featimp_list_list[r][v])
            df_tmp['omics'] = np.ones(df_tmp.shape[0], dtype=int) * v
            df_featimp = pd.concat([df_featimp, df_tmp.copy(deep=True)], ignore_index=True)

    df_featimp_top = df_featimp.groupby(['feat_name', 'omics'])['imp'].sum().reset_index()
    df_featimp_top = df_featimp_top.sort_values(by='imp', ascending=False).iloc[:topn]

    print('{:}\t{:}'.format('Rank', 'Feature name'))
    for i in range(len(df_featimp_top)):
        print('{:}\t{:}'.format(i + 1, df_featimp_top.iloc[i]['feat_name']))


