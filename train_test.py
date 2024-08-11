""" Training and testing of the model
"""
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn.functional as F
from models import init_model_dict, init_optim
from utils import one_hot_tensor, cal_sample_weight, gen_adj_mat_tensor, gen_test_adj_mat_tensor, cal_adj_mat_parameter

cuda = True if torch.cuda.is_available() else False
relu = torch.nn.ReLU()

def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)
    return torch.mean((A + B))

def DS_Combin(K, alpha):
    """
    :param alpha: All Dirichlet distribution parameters.
    :return: Combined Dirichlet distribution parameters.
    """
    def DS_Combin_two(alpha1, alpha2):
        """
        :param alpha1: Dirichlet distribution parameters of view 1
        :param alpha2: Dirichlet distribution parameters of view 2
        :return: Combined Dirichlet distribution parameters
        """
        alpha = dict()
        alpha[0], alpha[1] = alpha1, alpha2
        b, S, E, u = dict(), dict(), dict(), dict()
        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v]-1
            b[v] = E[v]/(S[v].expand(E[v].shape))
            u[v] = K/S[v]

        # b^0 @ b^(0+1)
        bb = torch.bmm(b[0].view(-1, K, 1), b[1].view(-1, 1, K))
        # b^0 * u^1
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        # b^1 * u^0
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)
        # calculate C
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        # bb_diag1 = torch.diag(torch.mm(b[v], torch.transpose(b[v+1], 0, 1)))
        C = bb_sum - bb_diag

        # calculate b^a
        b_a = (torch.mul(b[0], b[1]) + bu + ub)/((1-C).view(-1, 1).expand(b[0].shape))
        # calculate u^a
        u_a = torch.mul(u[0], u[1])/((1-C).view(-1, 1).expand(u[0].shape))

        # calculate new S
        S_a = K / u_a
        # calculate new e_k
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        return alpha_a

    for v in range(len(alpha)-1):
        if v==0:
            alpha_a = DS_Combin_two(alpha[0], alpha[1])
        else:
            alpha_a = DS_Combin_two(alpha_a, alpha[v+1])
    return alpha_a


def prepare_trte_data(data_folder, view_list):
    num_view = len(view_list)
    labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',')
    labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=',')
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)
    data_tr_list = []
    data_te_list = []
    for i in view_list:
        data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_tr.csv"), delimiter=','))
        data_te_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_te.csv"), delimiter=','))
    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]
    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
    data_tensor_list = []
    for i in range(len(data_mat_list)):
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].cuda()
    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["te"] = list(range(num_tr, (num_tr+num_te)))
    data_train_list = []
    data_all_list = []
    for i in range(len(data_tensor_list)):
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        data_all_list.append(torch.cat((data_tensor_list[i][idx_dict["tr"]].clone(),
                                       data_tensor_list[i][idx_dict["te"]].clone()),0))
    labels = np.concatenate((labels_tr, labels_te))
    
    return data_train_list, data_all_list, idx_dict, labels


def gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter):
    adj_metric = "cosine" # cosine distance
    adj_train_list = []
    adj_test_list = []
    for i in range(len(data_tr_list)):
        adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_tr_list[i], adj_metric)
        adj_train_list.append(gen_adj_mat_tensor(data_tr_list[i], adj_parameter_adaptive, adj_metric))
        adj_test_list.append(gen_test_adj_mat_tensor(data_trte_list[i], trte_idx, adj_parameter_adaptive, adj_metric))
    
    return adj_train_list, adj_test_list



def train_epoch(i_epoch, data_list, adj_list, label, one_hot_label, sample_weight, model_dict, optim_dict):
    loss_dict = {}
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    # Set models to training mode
    for m in model_dict:
        model_dict[m].train()

    num_view = len(data_list)

    # Train each view individually
    for i in range(num_view):
        optimizer = optim_dict["C{:}".format(i + 1)]
        optimizer.zero_grad()
        ci = model_dict["C{:}".format(i + 1)](model_dict["E{:}".format(i + 1)](data_list[i], adj_list[i]))
        ci_loss = torch.mean(torch.mul(criterion(ci, label), sample_weight))
        ci_loss.backward()
        optimizer.step()
        loss_dict["C{:}".format(i + 1)] = ci_loss.detach().cpu().numpy().item()

    # Additional processing for multiple views
    if num_view >= 2:
        alpahi_list = []
        K = one_hot_label.shape[1]

        for i in range(num_view):
            ei = relu(model_dict["C{:}".format(i + 1)](model_dict["E{:}".format(i + 1)](data_list[i], adj_list[i])))
            alpahi = ei + 1
            alpahi_list.append(alpahi)

        alpha_a = DS_Combin(K, alpahi_list)
        c_loss = sum(ce_loss(label, alpahi, K, i_epoch, 10) for alpahi in alpahi_list)
        c_loss += ce_loss(label, alpha_a, K, i_epoch, 10)

        # Zero the gradients for the combined loss update
        for i in range(1, num_view + 1):
            optim_dict["C{:}".format(i)].zero_grad()

        c_loss.backward()

        # Update the optimizers
        for i in range(1, num_view + 1):
            optim_dict["C{:}".format(i)].step()
            loss_dict["C{:}".format(i)] = c_loss.detach().cpu().numpy().item()

    return loss_dict

def test_epoch(one_hot_label, data_list, adj_list, te_idx, model_dict):
    for m in model_dict:
        model_dict[m].eval()
    num_view = len(data_list)
    alpahi_list = []
    K = one_hot_label.shape[1]

    if num_view >= 2:
        for i in range(num_view):
            ei = relu(model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i], adj_list[i])))
            alpahi = ei + 1
            alpahi_list.append(alpahi)
        alpha_a = DS_Combin(K, alpahi_list)
    else:
        # 如果只有一个视图，直接使用该视图的结果
        ei = relu(model_dict["C{:}".format(1)](model_dict["E{:}".format(1)](data_list[0], adj_list[0])))
        alpahi = ei + 1
        alpahi_list.append(alpahi)
        alpha_a = alpahi  # 只有一个视图的情况下 alpha_a 就是 alpahi

    v1 = alpahi_list[0][te_idx, :]
    v2 = alpahi_list[1][te_idx, :] if num_view > 1 else None
    v3 = alpahi_list[2][te_idx, :] if num_view > 2 else None
    all = alpha_a[te_idx, :]

    return v1, v2, v3, all


def train_test(data_folder, view_list, num_class,
               lr_e_pretrain, lr_e,num_epoch_pretrain,
               num_epoch):
    test_interval = 50
    num_view = len(view_list)

    # Set parameters based on dataset
    if data_folder == 'ROSMAP':
        adj_parameter = 2
        dim_he_list = [200, 200, 100]
    elif data_folder == 'BRCA':
        adj_parameter = 10
        dim_he_list = [400, 400, 200]

    # Prepare data
    data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_trte_data(data_folder, view_list)
    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    labels_te_tensor = torch.LongTensor(labels_trte[trte_idx["te"]])
    onehot_labels_te_tensor = one_hot_tensor(labels_te_tensor, num_class)
    sample_weight_tr = cal_sample_weight(labels_trte[trte_idx["tr"]], num_class)
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)

    # Move to GPU if available
    if torch.cuda.is_available():
        labels_tr_tensor = labels_tr_tensor.cuda()
        onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
        sample_weight_tr = sample_weight_tr.cuda()

    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)
    dim_list = [x.shape[1] for x in data_tr_list]
    model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list)

    for m in model_dict:
        if torch.cuda.is_available():
            model_dict[m].cuda()

    print("\nPretrain GCNs...")
    optim_dict = init_optim(num_view, model_dict, lr_e_pretrain)
    for epoch in range(num_epoch_pretrain):
        train_epoch(epoch, data_tr_list, adj_tr_list, labels_tr_tensor,
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict)

    print("\nTraining...")
    optim_dict = init_optim(num_view, model_dict, lr_e)
    best_acc, best_f1, best_auc, best_f1_weighted, best_f1_macro = 0, 0, 0, 0, 0

    for epoch in range(num_epoch + 1):
        train_epoch(epoch, data_tr_list, adj_tr_list, labels_tr_tensor,
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict)

        if epoch % test_interval == 0:
            v1, v2, v3, all = test_epoch(onehot_labels_te_tensor, data_trte_list, adj_te_list, trte_idx["te"],
                                         model_dict)
            print("\nTest: Epoch {:d}".format(epoch))

            v1_pred = v1.argmax(dim=1).cpu().detach().numpy()
            v2_pred = v2.argmax(dim=1).cpu().detach().numpy()
            v3_pred = v3.argmax(dim=1).cpu().detach().numpy()
            all_pred = all.argmax(dim=1).cpu().detach().numpy()

            if num_class == 2:
                for i, (pred, v) in enumerate(zip([v1_pred, v2_pred, v3_pred, all_pred], [v1, v2, v3, all]), start=1):
                    acc = accuracy_score(labels_trte[trte_idx["te"]], pred)
                    f1 = f1_score(labels_trte[trte_idx["te"]], pred)
                    auc = roc_auc_score(labels_trte[trte_idx["te"]], F.softmax(v, dim=1).data.cpu().numpy()[:, 1])
                    print(f"################ V{i} ###############")
                    print(f"Test ACC: {acc:.3f}")
                    print(f"Test F1: {f1:.3f}")
                    print(f"Test AUC: {auc:.3f}")
                    if i == 4 and auc > best_auc:
                        best_acc, best_f1, best_auc = acc, f1, auc
            else:
                for i, pred in enumerate([v1_pred, v2_pred, v3_pred, all_pred], start=1):
                    acc = accuracy_score(labels_trte[trte_idx["te"]], pred)
                    f1_weighted = f1_score(labels_trte[trte_idx["te"]], pred, average='weighted')
                    f1_macro = f1_score(labels_trte[trte_idx["te"]], pred, average='macro')
                    print(f"################ V{i} ###############")
                    print(f"Test ACC: {acc:.3f}")
                    print(f"Test F1 weighted: {f1_weighted:.3f}")
                    print(f"Test F1 macro: {f1_macro:.3f}")
                    if i == 4 and f1_macro > best_f1_macro:
                        best_acc = acc
                        best_f1_weighted = f1_weighted
                        best_f1_macro = f1_macro

    print()
    print(f"Best Test ACC: {best_acc:.3f}")
    print(f"Best Test F1: {best_f1:.3f}")
    print(f"Best Test AUC: {best_auc:.3f}")
    print(f"Best Test F1 weighted: {best_f1_weighted:.3f}")
    print(f"Best Test F1 macro: {best_f1_macro:.3f}")



