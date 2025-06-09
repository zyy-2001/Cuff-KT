import os
import argparse
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
import yaml
from data_loaders import (
    MostRecentQuestionSkillDataset,
    MostEarlyQuestionSkillDataset,
    SelectQuestionSkillDataset,
    PreprocessQuestionSkillDataset,
    ATDKTDatasetWrapper,
    DimKTDatasetWrapper,
    CounterDatasetWrapper,
    CuffDatasetWrapper,
)
from models.dkt import DKT
from models.atdkt import ATDKT
from models.dimkt import DIMKT
from models.stablekt import stableKT
from models.diskt import DisKT
from models.dkvmn import DKVMN
from train import model_train
from sklearn.model_selection import KFold
from datetime import datetime, timedelta
from utils.config import ConfigNode as CN
from utils.file_io import PathManager
from tqdm import tqdm
from collections import defaultdict
from torch.distributions import Categorical
import heapq
from scipy.stats import entropy





def main(config):
    accelerator = Accelerator()
    device = accelerator.device

    model_name = config.model_name
    dataset_path = config.dataset_path
    data_name = config.data_name
    seed = config.seed
    method = config.method
    control = config.control
    ratio = config.ratio
    exp = config.exp
    rank = config.rank
    convert = config.convert
    type = config.type

    np.random.seed(seed)
    torch.manual_seed(seed)

    
    train_config = config.train_config
    checkpoint_dir = config.checkpoint_dir

    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    ckpt_path = os.path.join(checkpoint_dir, model_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    ckpt_path = os.path.join(ckpt_path, data_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    batch_size = train_config.batch_size
    eval_batch_size = train_config.eval_batch_size
    learning_rate = train_config.learning_rate
    optimizer = train_config.optimizer
    seq_len = train_config.seq_len

    if train_config.sequence_option == "recent":  # the most recent N interactions
        dataset = MostRecentQuestionSkillDataset
    elif train_config.sequence_option == "early":  # the most early N interactions
        dataset = MostEarlyQuestionSkillDataset
    else:
        raise NotImplementedError("sequence option is not valid")

    if exp == 'intra':
        dataset = SelectQuestionSkillDataset


    test_aucs, test_accs, test_rmses = [], [], []


    df_path = os.path.join(os.path.join(dataset_path, data_name), "preprocessed_df.csv")
    df = pd.read_csv(df_path, sep="\t")
    if model_name == 'dimkt':
        questions_difficult_path = os.path.join(os.path.join(dataset_path, data_name), "questions_difficult_100.csv")
        skills_difficult_path = os.path.join(os.path.join(dataset_path, data_name), "skills_difficult_100.csv")
        def difficult_compute(df, difficult_path, tag, diff_level=100):
            sd = {}
            df = df.reset_index(drop=True)
            set_tags = set(np.array(df[tag]))
            from tqdm import tqdm
            for i in tqdm(set_tags):
                count = 0
                idx = df[(df[tag] == i)].index.tolist()
                tmp_data = df.iloc[idx]
                correct_1 = tmp_data['correct']
                if len(idx) < 30:
                    sd[i] = 1
                    continue
                else:
                    for j in np.array(correct_1):
                        count += j
                    if count == 0:
                        sd[i] = 1
                        continue
                    else:
                        avg = int((count/len(correct_1))*diff_level)+1
                        sd[i] = avg
            with open(difficult_path,'w',newline='',encoding='UTF8') as f:
                import csv
                writer = csv.writer(f)
                writer.writerow(sd.keys())
                writer.writerow(sd.values())
            return

        if not os.path.exists(questions_difficult_path):
            difficult_compute(df, questions_difficult_path, 'item_id')
        if not os.path.exists(skills_difficult_path):
            difficult_compute(df, skills_difficult_path, 'skill_id')

        sds = {}
        qds = {}
        import csv
        with open(skills_difficult_path,'r',encoding="UTF8") as f:
            reader = csv.reader(f)
            sds_keys = next(reader)
            sds_vals = next(reader)
            for i in range(len(sds_keys)):
                sds[int(sds_keys[i])] = int(sds_vals[i])
        with open(questions_difficult_path,'r',encoding="UTF8") as f:
            reader = csv.reader(f)
            qds_keys = next(reader)
            qds_vals = next(reader)
            for i in range(len(qds_keys)):
                qds[int(qds_keys[i])] = int(qds_vals[i])

    print("skill_min", df["skill_id"].min())
    users = df["user_id"].unique()
    df["skill_id"] += 1  # zero for padding
    df["item_id"] += 1  # zero for padding
    num_skills = df["skill_id"].max() + 1
    num_questions = df["item_id"].max() + 1

    np.random.shuffle(users)

    print("MODEL", model_name)
    print(dataset)
    # if data_name in ["statics", "assistments15"]:
    #     num_questions = 0
    def jensen_shannon_divergence(p, q):
            p = np.asarray(p)
            q = np.asarray(q)
            m = 0.5 * (p + q)
            
            kl_p_m = entropy(p, m)
            kl_q_m = entropy(q, m)
            js = 0.5 * (kl_p_m + kl_q_m)
            
            return js
    def kl_divergence(dict1, dict2):
        all_keys = sorted(set(dict1.keys()) | set(dict2.keys()))
        
        p = np.array([dict1.get(k, 1) for k in all_keys])
        q = np.array([dict2.get(k, 1) for k in all_keys])
        
        p = p / np.sum(p)
        q = q / np.sum(q)
        return jensen_shannon_divergence(p, q)

    if exp == 'inter':
        stus_kl_path = os.path.join(os.path.join(dataset_path, data_name), "stus_kl.csv")
        if not os.path.exists(stus_kl_path):
            # preprocess, giving each student a KL divergence to indicate changes in interest
            model_pre_config = config.dkt_config
            model_pre = DKT(num_skills, "none", 0, **model_pre_config)
            stus_kl = {}
            dataset_pre = PreprocessQuestionSkillDataset(df, seq_len, num_skills, num_questions)
            loader_pre = accelerator.prepare(
                DataLoader(dataset_pre, batch_size=batch_size)
            )
            n_gpu = torch.cuda.device_count()
            if n_gpu > 1:
                model_pre = torch.nn.DataParallel(model_pre).to(device)
            else:
                model_pre = model_pre.to(device)

            if optimizer == "sgd":
                opt_pre = SGD(model_pre.parameters(), learning_rate, momentum=0.9)
            elif optimizer == "adam":
                opt_pre = Adam(model_pre.parameters(), learning_rate, weight_decay=train_config.wl)

            model_pre, opt_pre = accelerator.prepare(model_pre, opt_pre)
            for i in range(100):
                for batch in tqdm(loader_pre):
                    opt_pre.zero_grad()
                    model_pre.train()
                    out_dict = model_pre(batch)
                    if n_gpu > 1:
                        loss, token_cnt, label_sum = model_pre.module.loss(batch, out_dict)
                    else:
                        loss, token_cnt, label_sum = model_pre.loss(batch, out_dict)
                    accelerator.backward(loss)

                    if train_config["max_grad_norm"] > 0.0:
                        torch.nn.utils.clip_grad_norm_(
                            model_pre.parameters(), max_norm=train_config["max_grad_norm"]
                        )
                    opt_pre.step()
            with torch.no_grad():
                for batch in tqdm(loader_pre, 'preprocess...'):
                    model_pre.eval()
                    out_dict = model_pre(batch)
                    stus = batch["stu_ids"]
                    state = out_dict['state']
                    assert state.size(0) == len(stus)
                    sm = out_dict["true"] > -1

                    last_idxs = sm.sum(dim=1) - 1
                    for i in range(len(stus)):
                        sid = stus[i]
                        lst_i = last_idxs[i]
                        state_s  = state[i]
                        mid_i = torch.div(0 + lst_i, 2, rounding_mode='floor') 
                        mid_state = state_s[mid_i]
                        lst_state = state_s[lst_i]
                        mid_state_norm = mid_state / mid_state.sum()
                        lst_state_norm = lst_state / lst_state.sum()
                        mid_dist = Categorical(probs=mid_state_norm)
                        lst_dist = Categorical(probs=lst_state_norm)
                        kl_div = torch.distributions.kl_divergence(mid_dist, lst_dist)
                        stus_kl[sid.item()] = float(kl_div.item())

            with open(stus_kl_path,'w',newline='',encoding='UTF8') as f:
                import csv
                writer = csv.writer(f)
                writer.writerow(stus_kl.keys())
                writer.writerow(stus_kl.values())
        import csv
        stus_kl = defaultdict(int)
        with open(stus_kl_path,'r',encoding="UTF8") as f:
            reader = csv.reader(f)
            stus = next(reader)
            stus_kls = next(reader)
            for i in range(len(stus)):
                stus_kl[int(stus[i])] = float(stus_kls[i])
        stus_ordered = heapq.nsmallest(len(stus_kl), stus_kl, key=stus_kl.get)
        users = stus_ordered


    for seed in range(5):
        np.random.seed(seed)
        torch.manual_seed(seed)
        model_base, opt_base = None, None
        if model_name == 'dkt':
            model_config = config.dkt_config
            model = DKT(num_skills, method, rank, **model_config)
            if control != 'none':
                model_base = DKT(num_skills, 'none', rank, **model_config)
        elif model_name == 'atdkt':
            model_config = config.atdkt_config
            model = ATDKT(num_skills, num_questions, method, rank, **model_config)
            if control != 'none':
                model_base = ATDKT(num_skills, num_questions, 'none', rank, **model_config)
        elif model_name == 'dimkt':
            model_config = config.dimkt_config
            model = DIMKT(convert, num_skills, num_questions, method, rank, **model_config)
            if control != 'none':
                model_base = DIMKT(convert, num_skills, num_questions, 'none', rank, **model_config)
        elif model_name == 'stablekt':
            model_config = config.stablekt_config
            model = stableKT(convert, num_skills, num_questions, method, rank, **model_config)
            if control != 'none':
                model_base = stableKT(convert, num_skills, num_questions, "none", rank, **model_config)
        elif model_name == 'diskt':
            model_config = config.diskt_config
            model = DisKT(convert, num_skills, num_questions, method, rank, seq_len, **model_config)
            if control != 'none':
                model_base = DisKT(convert, num_skills, num_questions, "none", rank, **model_config)
        elif model_name == 'dkvmn':
            model_config = config.dkvmn_config
            model = DKVMN(convert, num_skills, method, rank, **model_config)
            if control != 'none':
                model_base = DKVMN(convert, num_skills, "none", rank, **model_config)
        
        if method == 'cuff' and control == 'none':
            print(f'cuff_total_params: {sum(p.numel() for p in model.dpg.parameters())/1000}K')
            # import sys
            # sys.exit()

        tune_train_loader, tune_valid_loader = None, None
        if exp == 'inter':
            offset = int(len(users) * 0.7)
            test_offset = int(len(users) * 0.9)
            train_users = users[: offset]
            np.random.shuffle(train_users)
            valid_users = users[offset: test_offset]
            np.random.shuffle(valid_users)
            test_users = users[test_offset: ]
            np.random.shuffle(test_users)


            train_df = df[df["user_id"].isin(train_users)]
            valid_df = df[df["user_id"].isin(valid_users)]
            test_df = df[df["user_id"].isin(test_users)]

            if method in ['fft', 'adapter', 'lora', 'bitfit', 'cuff+']:
                tune_offset = int(len(valid_users) * 0.8)
                tune_train_users = valid_users[: tune_offset]
                tune_valid_users = valid_users[tune_offset: ]
                tune_train_df = df[df["user_id"].isin(tune_train_users)]
                tune_valid_df = df[df["user_id"].isin(tune_valid_users)]
                tune_train_dataset = dataset(tune_train_df, seq_len, num_skills, num_questions)
                tune_valid_dataset = dataset(tune_valid_df, seq_len, num_skills, num_questions)

            train_dataset = dataset(train_df, seq_len, num_skills, num_questions)
            valid_dataset = dataset(valid_df, seq_len, num_skills, num_questions)
            test_dataset = dataset(test_df, seq_len, num_skills, num_questions)

        elif exp == 'intra':
            train_dataset = dataset(df, seq_len, num_skills, num_questions, 0, 0.7)
            valid_dataset = dataset(df, seq_len, num_skills, num_questions, 0.7, 0.9)
            test_dataset = dataset(df, seq_len, num_skills, num_questions, 0.9, 1.0)
            tune_train_dataset = dataset(df, seq_len, num_skills, num_questions, 0.7, 0.8)
            tune_valid_dataset = dataset(df, seq_len, num_skills, num_questions, 0.8, 0.9)

        
        if exp == 'inter':
            print("train_ids", len(train_users))
            print("valid_ids", len(valid_users))
            print("test_ids", len(test_users))

        if "atdkt" in model_name: # atdkt
            train_loader = accelerator.prepare(
                DataLoader(
                    ATDKTDatasetWrapper(
                        train_dataset,
                        seq_len,
                        method,
                        control,
                    ),
                    batch_size=batch_size,
                )
            )

            valid_loader = accelerator.prepare(
                DataLoader(
                    ATDKTDatasetWrapper(
                        valid_dataset,
                        seq_len,
                        method,
                        control,
                    ),
                    batch_size=eval_batch_size,
                )
            )
            if method in ['fft', 'adapter', 'lora', 'bitfit', 'cuff+']:
                tune_train_loader = accelerator.prepare(
                    DataLoader(
                        ATDKTDatasetWrapper(
                            tune_train_dataset,
                            seq_len,
                            method,
                            control,
                        ),
                        batch_size=eval_batch_size,
                    )
                )
                tune_valid_loader = accelerator.prepare(
                    DataLoader(
                        ATDKTDatasetWrapper(
                            tune_valid_dataset,
                            seq_len,
                            method,
                            control,
                        ),
                        batch_size=eval_batch_size,
                    )
                )
                

            test_loader = accelerator.prepare(
                DataLoader(
                    ATDKTDatasetWrapper(
                        test_dataset,
                        seq_len,
                        method,
                        control,
                    ),
                    batch_size=eval_batch_size,
                )
            )
        elif "dis" in model_name:  # diskt
            train_loader = accelerator.prepare(
                DataLoader(
                    CounterDatasetWrapper(
                        train_dataset,
                        seq_len,
                        method,
                        control,
                    ),
                    batch_size=batch_size,
                )
            )

            valid_loader = accelerator.prepare(
                DataLoader(
                    CounterDatasetWrapper(
                        valid_dataset,
                        seq_len,
                        method,
                        control,
                    ),
                    batch_size=eval_batch_size,
                )
            )

            if method in ['fft', 'adapter', 'lora', 'bitfit', 'cuff+']:
                tune_train_loader = accelerator.prepare(
                    DataLoader(
                        CounterDatasetWrapper(
                            tune_train_dataset,
                            seq_len,
                            method,
                            control,
                        ),
                        batch_size=eval_batch_size,
                    )
                )
                tune_valid_loader = accelerator.prepare(
                    DataLoader(
                        CounterDatasetWrapper(
                            tune_valid_dataset,
                            seq_len,
                            method,
                            control,
                        ),
                        batch_size=eval_batch_size,
                    )
                )

            test_loader = accelerator.prepare(
                DataLoader(
                    CounterDatasetWrapper(
                        test_dataset,
                        seq_len,
                        method,
                        control,
                    ),
                    batch_size=eval_batch_size,
                )
            )
        elif "dimkt" in model_name: # dimkt
            train_loader = accelerator.prepare(
                DataLoader(
                    DimKTDatasetWrapper(
                        train_dataset,
                        seq_len,
                        sds,
                        qds,
                        method,
                        control,
                    ),
                    batch_size=batch_size,
                )
            )

            valid_loader = accelerator.prepare(
                DataLoader(
                    DimKTDatasetWrapper(
                        valid_dataset,
                        seq_len,
                        sds,
                        qds,
                        method,
                        control,
                    ),
                    batch_size=eval_batch_size,
                )
            )

            if method in ['fft', 'adapter', 'lora', 'bitfit', 'cuff+']:
                tune_train_loader = accelerator.prepare(
                    DataLoader(
                        DimKTDatasetWrapper(
                            tune_train_dataset,
                            seq_len,
                            sds,
                            qds,
                            method,
                            control,
                        ),
                        batch_size=eval_batch_size,
                    )
                )
                tune_valid_loader = accelerator.prepare(
                    DataLoader(
                        DimKTDatasetWrapper(
                            tune_valid_dataset,
                            seq_len,
                            sds,
                            qds,
                            method,
                            control,
                        ),
                        batch_size=eval_batch_size,
                    )
                )

            test_loader = accelerator.prepare(
                DataLoader(
                    DimKTDatasetWrapper(
                        test_dataset,
                        seq_len,
                        sds,
                        qds,
                        method,
                        control,
                    ),
                    batch_size=eval_batch_size,
                )
            )
        else:
            if method != 'cuff' and method != 'cuff+':
                train_loader = accelerator.prepare(
                    DataLoader(train_dataset, batch_size=batch_size)
                )

                valid_loader = accelerator.prepare(
                    DataLoader(valid_dataset, batch_size=eval_batch_size)
                )

                if method in ['fft', 'adapter', 'lora', 'bitfit', 'cuff+']:
                    tune_train_loader = accelerator.prepare(
                        DataLoader(tune_train_dataset, batch_size=eval_batch_size)
                    )
                    tune_valid_loader = accelerator.prepare(
                        DataLoader(tune_valid_dataset, batch_size=eval_batch_size)
                    )

                test_loader = accelerator.prepare(
                    DataLoader(test_dataset, batch_size=eval_batch_size)
                )
            else:
                train_loader = accelerator.prepare(
                    DataLoader(
                        CuffDatasetWrapper(
                            train_dataset,
                            seq_len,
                            control,
                        ),
                        batch_size=batch_size,
                    )
                )

                valid_loader = accelerator.prepare(
                    DataLoader(
                        CuffDatasetWrapper(
                            valid_dataset,
                            seq_len,
                            control,
                        ),
                        batch_size=eval_batch_size,
                    )
                )

                test_loader = accelerator.prepare(
                    DataLoader(
                        CuffDatasetWrapper(
                            test_dataset,
                            seq_len,
                            control,
                        ),
                        batch_size=eval_batch_size,
                    )
                )
                if method == 'cuff+':
                    tune_train_loader = accelerator.prepare(
                        DataLoader(
                            CuffDatasetWrapper(
                                tune_train_dataset,
                                seq_len,
                                control,
                            ),
                            batch_size=batch_size,
                        )
                    )
                    tune_valid_loader = accelerator.prepare(
                        DataLoader(
                            CuffDatasetWrapper(
                                tune_valid_dataset,
                                seq_len,
                                control,
                            ),
                            batch_size=batch_size,
                        )
                    )


        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            model = torch.nn.DataParallel(model).to(device)
            if control != 'none':
                model_base = torch.nn.DataParallel(model_base).to(device)
        else:
            model = model.to(device)
            if control != 'none':
                model_base = model_base.to(device)

        if optimizer == "sgd":
            opt = SGD(model.parameters(), learning_rate, momentum=0.9)
            if control != 'none':
                opt_base = SGD(model_base.parameters(), learning_rate, momentum=0.9)
        elif optimizer == "adam":
            opt = Adam(model.parameters(), learning_rate, weight_decay=train_config.wl)
            if control != 'none':
                opt_base = Adam(model_base.parameters(), learning_rate, weight_decay=train_config.wl)

        # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        # params = sum([np.prod(p.size()) for p in model_parameters])
        # print(f'params: {params}')
        # import sys
        # sys.exit()

        model, opt = accelerator.prepare(model, opt)
        if control != 'none':
            model_base, opt_base = accelerator.prepare(model_base, opt_base)


        test_auc, test_acc, test_rmse = model_train(
            seed,
            model,
            accelerator,
            opt,
            train_loader,
            valid_loader,
            test_loader,
            config,
            n_gpu,
            exp,
            method,
            control,
            ratio,
            model_base,
            opt_base,
            tune_train_loader,
            tune_valid_loader,
            type
        )

        test_aucs.append(test_auc)
        test_accs.append(test_acc)
        test_rmses.append(test_rmse)


    test_auc = np.mean(test_aucs)
    test_auc_std = np.std(test_aucs)
    test_acc = np.mean(test_accs)
    test_acc_std = np.std(test_accs)
    test_rmse = np.mean(test_rmses)
    test_rmse_std = np.std(test_rmses)

    now = (datetime.now() + timedelta(hours=9)).strftime("%Y%m%d-%H%M%S")  # KST time

    log_out_path = os.path.join(
        os.path.join("logs", "5-fold-cv", "{}".format(exp), "{}".format(data_name), "{}".format(method), "rank{}".format(rank))
    )
    if control != 'none':
        log_out_path = os.path.join(
            os.path.join("logs", "dkt-control", "{}".format(exp), "{}".format(data_name), "{}".format(method), "{}".format(control), "{}".format(ratio), "type{}".format(type))
        )
    os.makedirs(log_out_path, exist_ok=True)
    with open(os.path.join(log_out_path, "{}-{}".format(model_name, now)), "w") as f:
        f.write("AUC\tACC\tRMSE\tAUC_std\tACC_std\tRMSE_std\n")
        f.write("{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\n".format(test_auc, test_acc, test_rmse, test_auc_std, test_acc_std, test_rmse_std))
        f.write("AUC_ALL\n")
        f.write(",".join([str(auc) for auc in test_aucs])+"\n")
        f.write("ACC_ALL\n")
        f.write(",".join([str(auc) for auc in test_accs])+"\n")
        f.write("RMSE_ALL\n")
        f.write(",".join([str(auc) for auc in test_rmses])+"\n")

    print("\n5-fold CV Result")
    print("AUC\tACC\tRMSE")
    print("{:.5f}\t{:.5f}\t{:.5f}".format(test_auc, test_acc, test_rmse))


if __name__ == "__main__":
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="dkt",
        help="The name of the model to train. \
            The possible models are in [dkt, atdkt, dimkt, stablekt, diskt, dkvmn]. \
            The default model is dkt.",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="assistments09",
        help="The name of the dataset to use in training.",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.2, help="dropout probability"
    )
    parser.add_argument(
        "--batch_size", type=float, default=512, help="train batch size"
    )
    parser.add_argument(
        "--embedding_size", type=int, default=64, help="embedding size"
    )
    parser.add_argument(
        "--method", type=str, default='none', help="the possible methods are in [none, finetune(fft, lora, adapter, bitfit), cuff, cuff+]"
    )
    parser.add_argument(
        "--control", type=str, default='none', help="the possible control methods are in [none, cuff, pca, ecod, iforest, lof]"
    )
    parser.add_argument(
        "--ratio", type=float, default=0, help="the possible params. ratio are in [0, 0.2, 0.4, 0.6, 0.8, 1]"
    )
    parser.add_argument(
        "--exp", type=str, default='inter', help="Experiments are conducted either between student sequences (inter) or within student sequences (intra)"
    )
    parser.add_argument(
        "--rank", type=int, default=0, help="the rank of cuff"
    )
    parser.add_argument(
        "--convert", type=str2bool, default=False, help="Convert DIMKT's output settings to multi-concept output"
    )
    parser.add_argument(
        "--type", type=int, default=0, help="0 for default, 1 for w/o KL, 2 for w/o ZPD, 3 for w/o len"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--optimizer", type=str, default="adam", help="optimizer")
    args = parser.parse_args()
    assert args.model_name in ["dkt", "atdkt", "dimkt", "stablekt", "diskt", "dkvmn"]
    assert args.method in ["none", "fft", "lora", "adapter", "bitfit", "cuff", "cuff+"]
    assert args.control in ["none", "cuff", "pca", "ecod", "iforest", "lof"]
    assert args.exp in ["inter", "intra"]

    base_cfg_file = PathManager.open("configs/example.yaml", "r")
    base_cfg = yaml.safe_load(base_cfg_file)
    cfg = CN(base_cfg)
    cfg.set_new_allowed(True)
    cfg.model_name = args.model_name
    cfg.data_name = args.data_name
    cfg.method = args.method
    cfg.control = args.control
    cfg.ratio = args.ratio
    cfg.exp = args.exp
    cfg.rank = args.rank
    cfg.convert = args.convert
    if cfg.control != 'none':
        assert cfg.method == 'cuff', 'method must be cuff when control is not none'
    cfg.train_config.batch_size = int(args.batch_size)
    cfg.train_config.eval_batch_size = int(args.batch_size)
    cfg.train_config.learning_rate = args.lr
    cfg.train_config.optimizer = args.optimizer

    if args.model_name == 'dkt':  # dkt
        cfg.dkt_config.dropout = args.dropout
    elif args.model_name == 'atdkt':  # atdkt
        cfg.atdkt_config.dropout = args.dropout
    elif args.model_name == 'dimkt':  # dimkt 
        cfg.dimkt_config.dropout = args.dropout
    elif args.model_name == 'stablekt':  # stablekt
        cfg.stablekt_config.dropout = args.dropout
    elif args.model_name == 'diskt':  # diskt
        cfg.diskt_config.dropout = args.dropout

    cfg.type = args.type
    cfg.freeze()

    print(cfg)
    main(cfg)
