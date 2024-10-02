import torch.nn as nn

import pandas as pd
import numpy as np
import torch
import os
import glob

from datetime import datetime, timedelta
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

from pyod.models.lof import LOF
from pyod.models.ecod import ECOD
from pyod.models.pca import PCA
from pyod.models.iforest import IForest
import heapq
from torch.distributions import Categorical
import loralib as lora
from fvcore.nn import FlopCountAnalysis
import time
from torch.optim import SGD, Adam


def set_requires_grad(model, trainable=True):
    for param in model.parameters():
        param.requires_grad = trainable

class AdapterLayer(nn.Module):
    def __init__(self, input_dim, adapter_dim):
        super(AdapterLayer, self).__init__()
        self.down_project = nn.Linear(input_dim, adapter_dim)
        self.activation = nn.ReLU()
        self.up_project = nn.Linear(adapter_dim, input_dim)
    
    def forward(self, x):
        down_projected = self.down_project(x)
        activated = self.activation(down_projected)
        up_projected = self.up_project(activated)
        return up_projected + x 

def insert_adapter_layers(model, adapter_dim=4):
    for name, module in model.named_children():
        if isinstance(module, nn.TransformerEncoder) or isinstance(module, nn.TransformerEncoderLayer):
            continue
        if isinstance(module, nn.Linear):
            adapter_layer = AdapterLayer(module.out_features, adapter_dim)
            setattr(model, name, nn.Sequential(module, adapter_layer))
        else:
            insert_adapter_layers(module, adapter_dim)
    return model

def freeze_model_parameters(model):
    for param in model.parameters():
        param.requires_grad = False

def trainable_adapter_parameters(model):
    for name, module in model.named_modules():
        if isinstance(module, AdapterLayer):
            for param in module.parameters():
                param.requires_grad = True


def freeze_except_bias(model):
    for name, param in model.named_parameters():
        if "bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def model_train(
    fold,
    model,
    accelerator,
    opt,
    train_loader,
    valid_loader,
    test_loader,
    config,
    n_gpu,
    exp = 'inter',
    method = 'none',
    control = 'none',
    ratio = 0,
    model_base = None,
    opt_base = None,
    tune_train_loader = None,
    tune_valid_loader = None,
):
    train_losses = []
    avg_train_losses = []
    best_valid_auc = 0

    num_epochs = config["train_config"]["num_epochs"]
    model_name = config["model_name"]
    data_name = config["data_name"]
    train_config = config["train_config"]


    token_cnts = 0
    label_sums = 0


    for i in range(1, num_epochs + 1):
        for batch in tqdm(train_loader):
            opt.zero_grad()

            model.train()
            out_dict = model(batch)

            if n_gpu > 1:
                loss, token_cnt, label_sum = model.module.loss(batch, out_dict)
            else:
                loss, token_cnt, label_sum = model.loss(batch, out_dict)
            accelerator.backward(loss)

            token_cnts += token_cnt
            label_sums += label_sum

            if train_config["max_grad_norm"] > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=train_config["max_grad_norm"]
                )

            opt.step()
            train_losses.append(loss.item())

        print("token_cnts", token_cnts, "label_sums", label_sums)

        total_preds = []
        total_trues = []

        with torch.no_grad():
            for batch in valid_loader:
                model.eval()

                out_dict = model(batch)

                pred = out_dict["pred"].flatten()
                true = out_dict["true"].flatten()
                mask = true > -1
                true = true[mask]
                pred = pred[mask]

                total_preds.append(pred)
                total_trues.append(true)

            total_preds = torch.cat(total_preds).squeeze(-1).detach().cpu().numpy()
            total_trues = torch.cat(total_trues).squeeze(-1).detach().cpu().numpy()

        train_loss = np.average(train_losses)
        avg_train_losses.append(train_loss)

        valid_auc = roc_auc_score(y_true=total_trues, y_score=total_preds)

        path = os.path.join("saved_model", "{}".format(exp), "{}".format(method), "{}".format(control), "{}".format(ratio), model_name, data_name)
        
        if not os.path.isdir(path):
            os.makedirs(path)

        if valid_auc > best_valid_auc:

            path = os.path.join(
                os.path.join("saved_model", "{}".format(exp), "{}".format(method), "{}".format(control), "{}".format(ratio), model_name, data_name), "params_*"
            )
            for _path in glob.glob(path):
                os.remove(_path)
            best_valid_auc = valid_auc
            best_epoch = i
            torch.save(
                {"epoch": i, "model_state_dict": model.state_dict(),},
                os.path.join(
                    os.path.join("saved_model", "{}".format(exp), "{}".format(method), "{}".format(control), "{}".format(ratio), model_name, data_name),
                    "params_{}".format(str(best_epoch)),
                ),
            )
        if i - best_epoch > 10:
            break

        # clear lists to track next epochs
        train_losses = []

        total_preds, total_trues = [], []

        # evaluation on test dataset
        with torch.no_grad():
            for batch in test_loader:

                model.eval()

                out_dict = model(batch)
                
                pred = out_dict["pred"].flatten()
                true = out_dict["true"].flatten()
                mask = true > -1
                true = true[mask]
                pred = pred[mask]

                total_preds.append(pred)
                total_trues.append(true)

            total_preds = torch.cat(total_preds).squeeze(-1).detach().cpu().numpy()
            total_trues = torch.cat(total_trues).squeeze(-1).detach().cpu().numpy()

        test_auc = roc_auc_score(y_true=total_trues, y_score=total_preds)

        print(
            "Fold {}:\t Epoch {}\t\tTRAIN LOSS: {:.5f}\tVALID AUC: {:.5f}\tTEST AUC: {:.5f}".format(
                fold, i, train_loss, valid_auc, test_auc
            )
        )
    checkpoint = torch.load(
        os.path.join(
            os.path.join("saved_model", "{}".format(exp), "{}".format(method), "{}".format(control), "{}".format(ratio), model_name, data_name),
            "params_{}".format(str(best_epoch)),
        )
    )

    model.load_state_dict(checkpoint["model_state_dict"]) # pretrain if finetune

    if method in ['fft', 'adapter', 'bitfit', 'cuff+']:
        if method == 'adapter':
            model = insert_adapter_layers(model, adapter_dim=2)
            freeze_model_parameters(model)
            trainable_adapter_parameters(model)
        elif method == 'bitfit':
            freeze_except_bias(model)
        elif method == 'fft' or method == 'cuff+':
            set_requires_grad(model, trainable=True)

        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f'finetune_total_params: {params/1000}K')
        # import sys
        # sys.exit()
        
        start_time = time.perf_counter()
        optimizer = train_config.optimizer
        if optimizer == 'adam':
            opt = Adam(filter(lambda p: p.requires_grad, model.parameters()), train_config.learning_rate, weight_decay=train_config.wl)
        elif optimizer == 'sgd':
            opt = SGD(filter(lambda p: p.requires_grad, model.parameters()), train_config.learning_rate, momentum=0.9)


        train_losses = []
        avg_train_losses = []
        best_valid_auc = 0
        token_cnts, label_sum = 0, 0
        train_epoch = num_epochs
        for i in range(1, num_epochs + 1):
            for batch in tqdm(tune_train_loader):
                opt.zero_grad()

                model.train()
                out_dict = model(batch)

                if n_gpu > 1:
                    loss, token_cnt, label_sum = model.module.loss(batch, out_dict)
                else:
                    loss, token_cnt, label_sum = model.loss(batch, out_dict)
                accelerator.backward(loss)

                token_cnts += token_cnt
                label_sums += label_sum

                if train_config["max_grad_norm"] > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=train_config["max_grad_norm"]
                    )

                opt.step()
                train_losses.append(loss.item())

            print("token_cnts", token_cnts, "label_sums", label_sums)

            total_preds = []
            total_trues = []

            with torch.no_grad():
                for batch in tune_valid_loader:
                    model.eval()

                    out_dict = model(batch)
                    
                    pred = out_dict["pred"].flatten()
                    true = out_dict["true"].flatten()
                    mask = true > -1
                    true = true[mask]
                    pred = pred[mask]

                    total_preds.append(pred)
                    total_trues.append(true)

                total_preds = torch.cat(total_preds).squeeze(-1).detach().cpu().numpy()
                total_trues = torch.cat(total_trues).squeeze(-1).detach().cpu().numpy()

            train_loss = np.average(train_losses)
            avg_train_losses.append(train_loss)

            valid_auc = roc_auc_score(y_true=total_trues, y_score=total_preds)

            path = os.path.join("saved_model", "{}".format(exp), "{}".format(method), model_name+"pretrain", data_name)
            if not os.path.isdir(path):
                os.makedirs(path)

            if valid_auc > best_valid_auc:

                path = os.path.join(
                    os.path.join("saved_model", "{}".format(exp), "{}".format(method), model_name+"pretrain", data_name), "params_*"
                )
                for _path in glob.glob(path):
                    os.remove(_path)
                best_valid_auc = valid_auc
                best_epoch = i
                torch.save(
                    {"epoch": i, "model_state_dict": model.state_dict()},
                    os.path.join(
                        os.path.join("saved_model", "{}".format(exp), "{}".format(method), model_name+"pretrain", data_name),
                        "params_{}".format(str(best_epoch)),
                    ),
                )
            if i - best_epoch > 10:
                train_epoch = i
                break

        end_time = time.perf_counter()
        ft_time = round(((end_time - start_time) * 1000), 2) / train_epoch
        print(f'end_time: {end_time}')
        print(f'start_time: {start_time}')
        print(f"Fine-tuning time: {ft_time} ms")
        # import sys
        # sys.exit()
    
    if method in ['fft', 'adapter', 'bitfit', 'cuff+']:
        checkpoint = torch.load(
            os.path.join(
                os.path.join("saved_model", "{}".format(exp), "{}".format(method), model_name+"pretrain", data_name),
                "params_{}".format(str(best_epoch)),
            )
        )

        model.load_state_dict(checkpoint["model_state_dict"])

    if control != 'none':
        # model_base if control is not none
        train_losses = []
        avg_train_losses = []
        best_valid_auc = 0
        token_cnts = 0
        label_sums = 0

        for i in range(1, num_epochs + 1):
            for batch in tqdm(train_loader):
                opt_base.zero_grad()

                model_base.train()
                out_dict = model_base(batch)

                if n_gpu > 1:
                    loss, token_cnt, label_sum = model_base.module.loss(batch, out_dict)
                else:
                    loss, token_cnt, label_sum = model_base.loss(batch, out_dict)
                accelerator.backward(loss)

                token_cnts += token_cnt
                label_sums += label_sum

                if train_config["max_grad_norm"] > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        model_base.parameters(), max_norm=train_config["max_grad_norm"]
                    )

                opt_base.step()
                train_losses.append(loss.item())

            print("token_cnts", token_cnts, "label_sums", label_sums)

            total_preds = []
            total_trues = []

            with torch.no_grad():
                for batch in valid_loader:
                    model_base.eval()

                    out_dict = model_base(batch)
                    pred = out_dict["pred"].flatten()
                    true = out_dict["true"].flatten()
                    mask = true > -1
                    true = true[mask]
                    pred = pred[mask]

                    total_preds.append(pred)
                    total_trues.append(true)

                total_preds = torch.cat(total_preds).squeeze(-1).detach().cpu().numpy()
                total_trues = torch.cat(total_trues).squeeze(-1).detach().cpu().numpy()

            train_loss = np.average(train_losses)
            avg_train_losses.append(train_loss)

            valid_auc = roc_auc_score(y_true=total_trues, y_score=total_preds)

            path = os.path.join("saved_model", "{}".format(exp), "{}".format(control), "{}".format(ratio), model_name+'_base', data_name)
            if not os.path.isdir(path):
                os.makedirs(path)

            if valid_auc > best_valid_auc:

                path = os.path.join(
                    os.path.join("saved_model", "{}".format(exp), "{}".format(control), "{}".format(ratio), model_name+'_base', data_name), "params_*"
                )
                for _path in glob.glob(path):
                    os.remove(_path)
                best_valid_auc = valid_auc
                best_epoch = i
                torch.save(
                    {"epoch": i, "model_state_dict": model_base.state_dict(),},
                    os.path.join(
                        os.path.join("saved_model", "{}".format(exp), "{}".format(control), "{}".format(ratio), model_name+'_base', data_name),
                        "params_{}".format(str(best_epoch)),
                    ),
                )
            if i - best_epoch > 10:
                break

        checkpoint_base = torch.load(
            os.path.join(
                os.path.join("saved_model", "{}".format(exp), "{}".format(control), "{}".format(ratio), model_name+'_base', data_name),
                "params_{}".format(str(best_epoch)),
            )
        )

        model_base.load_state_dict(checkpoint_base["model_state_dict"])



    total_preds, total_trues = [], []
    if control != 'none':
        all_s = {}
        bid = 0
        max_batch = train_config["batch_size"]

    # start_time = time.perf_counter()
    # total_flops = 0
    with torch.no_grad():
        for batch in test_loader:

            model.eval()
            # if total_flops == 0:
            #     total_flops = FlopCountAnalysis(model, batch).total() / (batch['skills'].size(0))
            out_dict = model(batch)

            if control == 'cuff':
                correct_rates = batch["correct_rates"]
                sm = out_dict["true"] > -1
            pred = out_dict["pred"].flatten()
            true = out_dict["true"].flatten()
            mask = true > -1
            true = true[mask]
            if model_name != 'corekt':
                pred = pred[mask]
            

            if control != 'none':
                pred_s = out_dict["pred"]
                pred_s = pred_s.cpu()
                if control == 'lof':
                    clf = LOF()
                    clf.fit(pred_s)
                    scores = clf.decision_scores_
                elif control == 'ecod':
                    clf = ECOD()
                    clf.fit(pred_s)
                    scores = clf.decision_scores_
                elif control == 'pca':
                    clf = PCA()
                    clf.fit(pred_s)
                    scores = clf.decision_scores_
                elif control == 'iforest':
                    clf = IForest()
                    clf.fit(pred_s)
                    scores = clf.decision_scores_
                elif control == 'cuff':
                    state = out_dict["state"]
                    last_idxs = sm.sum(dim=1) - 1
     
                    for i in range(state.size(0)):
                        correct_rate = correct_rates[i]
                        sid = bid * max_batch + i
                        state_s = state[i]
                        len_i = last_idxs[i]
                        mid_i = torch.div(0 + len_i, 2, rounding_mode='floor') 
                        mid_state = state_s[mid_i]
                        lst_state = state_s[len_i]
                        mid_correct_rate = correct_rate[mid_i] + 1
                        lst_correct_rate = correct_rate[len_i] + 1
                        mid_state_norm = mid_state / mid_state.sum()
                        lst_state_norm = lst_state / lst_state.sum()
                        
                        mid_dist = Categorical(probs=mid_state_norm)
                        lst_dist = Categorical(probs=lst_state_norm)

                        KL = (torch.distributions.kl_divergence(mid_dist, lst_dist) + 1)

                        ZPO = (abs((lst_correct_rate - mid_correct_rate)) / mid_correct_rate + 1) * len_i
                        all_s[sid] = KL * ZPO

                if control != 'cuff':
                    for i in range(pred_s.size(0)):
                        sid = bid * max_batch + i
                        all_s[sid] = scores[i]
                bid += 1
            else:
                total_preds.append(pred)
                total_trues.append(true)
            

        if control == 'none':
            total_preds = torch.cat(total_preds).squeeze(-1).detach().cpu().numpy()
            total_trues = torch.cat(total_trues).squeeze(-1).detach().cpu().numpy()

    # end_time = time.perf_counter()
    # t_time = round(((end_time - start_time) * 1000), 2)
    # print(f'Test_time: {t_time} ms')
    # log_out_path = os.path.join(
    #         os.path.join("logs", "5-fold-cv", "{}".format(exp), "{}".format(data_name), "{}".format(method))
    #     )
    # os.makedirs(log_out_path, exist_ok=True)
    # now = (datetime.now() + timedelta(hours=9)).strftime("%Y%m%d-%H%M%S")
    # with open(os.path.join(log_out_path, "{}-{}".format(model_name, now)), "w") as f:
    #     f.write("Test time\n")
    #     f.write("{} ms".format(t_time))
    # print(f'total_flops: {total_flops/1e6}M')
    # import sys
    # sys.exit()
    if control == 'none':
        auc = roc_auc_score(y_true=total_trues, y_score=total_preds)
        acc = accuracy_score(y_true=total_trues >= 0.5, y_pred=total_preds >= 0.5)
        rmse = np.sqrt(mean_squared_error(y_true=total_trues, y_pred=total_preds))
    else:
        select_n = int(len(all_s) * ratio)
        stus = heapq.nlargest(select_n, all_s, key=all_s.get)
        stus.sort()
        bid = 0
        it = 0
        all_sids = []
        all_sids_base = []
        with torch.no_grad():
            for batch in test_loader:
                model.eval()
                model_base.eval()
                out_dict = model(batch)
                out_dict_base = model_base(batch)
                pred_s = out_dict["pred"]
                sids, sids_base = [], []
                for i in range(pred_s.size(0)):
                    sid = bid * max_batch + i
                    if it < len(stus) and sid == stus[it]:
                        sids.append(i)
                        all_sids.append(i)
                        it += 1
                    else:
                        sids_base.append(i)
                        all_sids_base.append(i)
                sids = torch.tensor(sids, dtype=torch.long)
                sids_base = torch.tensor(sids_base, dtype=torch.long)
                pred = out_dict["pred"][sids].flatten()
                true = out_dict["true"][sids].flatten()
                mask = true > -1
                true = true[mask]
                pred = pred[mask]
            
                total_preds.append(pred)
                total_trues.append(true)

                pred_base = out_dict_base["pred"][sids_base].flatten()
                true_base = out_dict_base["true"][sids_base].flatten()
                mask_base = true_base > -1
                true_base = true_base[mask_base]
                pred_base = pred_base[mask_base]
            
                total_preds.append(pred_base)
                total_trues.append(true_base)
                bid += 1

        total_preds = torch.cat(total_preds).squeeze(-1).detach().cpu().numpy()
        total_trues = torch.cat(total_trues).squeeze(-1).detach().cpu().numpy()
        auc = roc_auc_score(y_true=total_trues, y_score=total_preds)
        acc = accuracy_score(y_true=total_trues >= 0.5, y_pred=total_preds >= 0.5)
        rmse = np.sqrt(mean_squared_error(y_true=total_trues, y_pred=total_preds))
    print(
        "Best Model\tTEST AUC: {:.5f}\tTEST ACC: {:5f}\tTEST RMSE: {:5f}".format(
            auc, acc, rmse
        )
    )
    return auc, acc, rmse
