# This code is based on the following repositories:
#  1. https://github.com/theophilee/learner-performance-prediction/blob/master/prepare_data.py
#  2. https://github.com/THUwangcy/HawkesKT/blob/main/data/Preprocess.ipynb

from argparse import ArgumentParser
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy import sparse
import os
import pickle
import time


# Please specify your dataset Path
BASE_PATH = "./dataset"
np.random.seed(0)

def prepare_assistments(
    data_name: str, min_user_inter_num: int, remove_nan_skills: bool
):
    """
    Preprocess ASSISTments dataset

        :param data_name: (str) "assistments09", "assistments12", "assisments15", and "assistments17"
        :param min_user_inter_num: (int) Users whose number of interactions is less than min_user_inter_num will be removed
        :param remove_nan_skills: (bool) if True, remove interactions with no skill tage
        :param train_split: (float) proportion of data to use for training
        
        :output df: (pd.DataFrame) preprocssed ASSISTments dataset with user_id, item_id, timestamp, correct and unique skill features
        :output question_skill_rel: (csr_matrix) corresponding question-skill relationship matrix
    """
    data_path = os.path.join(BASE_PATH, data_name)
    df = pd.read_csv(os.path.join(data_path, "data.csv"), encoding="ISO-8859-1")

    # Only 2012 and 2017 versions have timestamps
    if data_name == "assistments09":
        # df = pd.read_csv(os.path.join(data_path, "skill_builder_data_corrected.csv"), encoding="ISO-8859-1")
        df = df.rename(columns={"problem_id": "item_id"})
        df["timestamp"] = np.zeros(len(df), dtype=np.int64)
    elif data_name == "assistments12":
        # df = pd.read_csv(os.path.join(data_path, "2012-2013-data-with-predictions-4-final.csv"), encoding="ISO-8859-1")
        df = df.rename(columns={"problem_id": "item_id"})
        df["timestamp"] = pd.to_datetime(df["start_time"])
        df["timestamp"] = df["timestamp"] - df["timestamp"].min()
        df["timestamp"] = (
            df["timestamp"].apply(lambda x: x.total_seconds()).astype(np.int64)
        )
    elif data_name == "assistments15":
        df = df.rename(columns={"sequence_id": "item_id"})
        df["skill_id"] = df["item_id"]
        df["timestamp"] = np.zeros(len(df), dtype=np.int64)
    elif data_name == "assistments17":
        df = df.rename(
            columns={
                "startTime": "timestamp",
                "studentId": "user_id",
                "problemId": "item_id",
                "skill": "skill_id",
            }
        )
        df["timestamp"] = df["timestamp"] - df["timestamp"].min()

    # Remove continuous outcomes
    df = df[df["correct"].isin([0, 1])]
    df["correct"] = df["correct"].astype(np.int32)

    # Filter nan skills
    if remove_nan_skills:
        df = df[~df["skill_id"].isnull()]
    else:
        df.loc[df["skill_id"].isnull(), "skill_id"] = -1

    # Filter too short sequences
    df = df.groupby("user_id").filter(lambda x: len(x) >= min_user_inter_num)

    df["user_id"] = np.unique(df["user_id"], return_inverse=True)[1]
    df["item_id"] = np.unique(df["item_id"], return_inverse=True)[1]
    df["skill_id"] = np.unique(df["skill_id"].astype(str), return_inverse=True)[1]
    if data_name != 'assistments15' and data_name != 'assistments17':
        with open(os.path.join(data_path, "skill_id_name"), "wb") as f:
            pickle.dump(dict(zip(df["skill_id"], df["skill_name"])), f)

    # Build Q-matrix
    Q_mat = np.zeros((len(df["item_id"].unique()), len(df["skill_id"].unique())))
    for item_id, skill_id in df[["item_id", "skill_id"]].values:
        Q_mat[item_id, skill_id] = 1

    # Remove row duplicates due to multiple skills for one item
    if data_name == "assistments09":
        df = df.drop_duplicates("order_id")
    elif data_name == "assistments17":
        df = df.drop_duplicates(["user_id", "timestamp"])

    print("# Users: {}".format(df["user_id"].nunique()))
    print("# Skills: {}".format(df["skill_id"].nunique()))
    print("# Items: {}".format(df["item_id"].nunique()))
    print("# Interactions: {}".format(len(df)))

    # Get unique skill id from combination of all skill ids
    unique_skill_ids = np.unique(Q_mat, axis=0, return_inverse=True)[1]
    df["skill_id"] = unique_skill_ids[df["item_id"]]

    print("# Preprocessed Skills: {}".format(df["skill_id"].nunique()))
    # Sort data temporally
    if data_name in ["assistments12", "assistments17"]:
        df.sort_values(by="timestamp", inplace=True)
    elif data_name == "assistments09":
        df.sort_values(by="order_id", inplace=True)
    elif data_name == "assistments15":
        df.sort_values(by="log_id", inplace=True)

    # Sort data by users, preserving temporal order for each user
    df = pd.concat([u_df for _, u_df in df.groupby("user_id")])
    df.to_csv(os.path.join(data_path, "original_df.csv"), sep="\t", index=False)

    df = df[["user_id", "item_id", "timestamp", "correct", "skill_id"]]

    df.reset_index(inplace=True, drop=True)

    # Save data
    with open(os.path.join(data_path, "question_skill_rel.pkl"), "wb") as f:
        pickle.dump(csr_matrix(Q_mat), f)

    sparse.save_npz(os.path.join(data_path, "q_mat.npz"), csr_matrix(Q_mat))
    df.to_csv(os.path.join(data_path, "preprocessed_df.csv"), sep="\t", index=False)


def prepare_comp(
    data_name: str, min_user_inter_num: int, remove_nan_skills: bool
):
    """
    Preprocess comp dataset

        :param data_name: (str) "prob", "linux", "comp", and "database"
        :param min_user_inter_num: (int) Users whose number of interactions is less than min_user_inter_num will be removed
        :param remove_nan_skills: (bool) if True, remove interactions with no skill tage
        :param train_split: (float) proportion of data to use for training
        
        :output df: (pd.DataFrame) preprocssed PATDisc dataset with user_id, item_id, timestamp, correct and unique skill features
        :output question_skill_rel: (csr_matrix) corresponding question-skill relationship matrix
    """
    data_path = os.path.join(BASE_PATH, data_name)
    df = pd.read_csv(os.path.join(data_path, "processed_data.csv"), encoding="ISO-8859-1")

    
    df = df.rename(
        columns={
            "create_at": "timestamp",
            "user_id_new": "user_id",
            "problem_id_new": "item_id",
            "skill_id_new": "skill_id",
            "score": "correct"
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["timestamp"] = df["timestamp"] - df["timestamp"].min()
    df["timestamp"] = (
        df["timestamp"].apply(lambda x: x.total_seconds()).astype(np.int64)
    )
    
    # Remove continuous outcomes
    df = df[df["correct"].isin([0, 1])]
    df["correct"] = df["correct"].astype(np.int32)

    # Filter nan skills
    if remove_nan_skills:
        df = df[~df["skill_id"].isnull()]
    else:
        df.loc[df["skill_id"].isnull(), "skill_id"] = -1

    user_wise_lst = list()
    for user, user_df in df.groupby("user_id"):
        if len(user_df) >= min_user_inter_num:
            filter_df = user_df.sort_values(by=["timestamp"])  # assure the sequence order
            user_wise_lst.append(filter_df)

    np.random.shuffle(user_wise_lst)
    user_list = user_wise_lst[:5000]  # sample 5000 students
    df = pd.concat(user_list).reset_index(drop=True)

    # # Filter too short sequences
    # df = df.groupby("user_id").filter(lambda x: len(x) >= min_user_inter_num)

    df["user_id"] = np.unique(df["user_id"], return_inverse=True)[1]
    df["item_id"] = np.unique(df["item_id"], return_inverse=True)[1]
    df["skill_id"] = np.unique(df["skill_id"].astype(str), return_inverse=True)[1]

    # Build Q-matrix
    Q_mat = np.zeros((len(df["item_id"].unique()), len(df["skill_id"].unique())))
    for item_id, skill_id in df[["item_id", "skill_id"]].values:
        Q_mat[item_id, skill_id] = 1


    print("# Users: {}".format(df["user_id"].nunique()))
    print("# Skills: {}".format(df["skill_id"].nunique()))
    print("# Items: {}".format(df["item_id"].nunique()))
    print("# Interactions: {}".format(len(df)))

    # Get unique skill id from combination of all skill ids
    unique_skill_ids = np.unique(Q_mat, axis=0, return_inverse=True)[1]
    df["skill_id"] = unique_skill_ids[df["item_id"]]

    print("# Preprocessed Skills: {}".format(df["skill_id"].nunique()))
    # Sort data temporally
    df.sort_values(by="timestamp", inplace=True)

    # Sort data by users, preserving temporal order for each user
    df = pd.concat([u_df for _, u_df in df.groupby("user_id")])
    df.to_csv(os.path.join(data_path, "original_df.csv"), sep="\t", index=False)

    df = df[["user_id", "item_id", "timestamp", "correct", "skill_id"]]

    df.reset_index(inplace=True, drop=True)

    # Save data
    with open(os.path.join(data_path, "question_skill_rel.pkl"), "wb") as f:
        pickle.dump(csr_matrix(Q_mat), f)

    sparse.save_npz(os.path.join(data_path, "q_mat.npz"), csr_matrix(Q_mat))
    df.to_csv(os.path.join(data_path, "preprocessed_df.csv"), sep="\t", index=False)



def prepare_xes3g5m(
    data_name: str, min_user_inter_num: int, remove_nan_skills: bool
):
    """
    Preprocess xes3g5m dataset

        :param data_name: (str) "xes3g5m"
        :param min_user_inter_num: (int) Users whose number of interactions is less than min_user_inter_num will be removed
        :param remove_nan_skills: (bool) if True, remove interactions with no skill tage
        :param train_split: (float) proportion of data to use for training
        
        :output df: (pd.DataFrame) preprocssed xes3g5m dataset with user_id, item_id, timestamp, correct and unique skill features
        :output question_skill_rel: (csr_matrix) corresponding question-skill relationship matrix
    """
    data_path = os.path.join(BASE_PATH, data_name)
    df_train = pd.read_csv(os.path.join(data_path, "train_valid_sequences.csv"))
    df_test = pd.read_csv(os.path.join(data_path, "test.csv"))
    columns = ["timestamps", "uid", "questions", "concepts", "responses"]
    df_train = df_train[columns]
    df_test = df_test[columns]
    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    df = df.reset_index(drop=True)
    
    df = df.rename(
        columns={
            "timestamps": "timestamp",
            "uid": "user_id",
            "questions": "item_id",
            "concepts": "skill_id",
            "responses": "correct"
        }
    )

    list_columns = ['timestamp', 'item_id', 'skill_id', 'correct']
    for col in list_columns:
        df[col] = df[col].str.split(',')
    # for col in ["timestamp", "user_id", "item_id", "skill_id", "correct"]:
    #     df[col] = df[col].apply(literal_eval)

    def expand_row(row):
        return pd.DataFrame({
            'user_id': [row['user_id']] * len(row['timestamp']),
            'timestamp': row['timestamp'],
            'item_id': row['item_id'],
            'skill_id': row['skill_id'],
            'correct': row['correct']
        })
    
    df = pd.concat(df.apply(expand_row, axis=1).tolist(), ignore_index=True)
    for col in list_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    df = df[df['timestamp'] != -1]

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["timestamp"] = df["timestamp"].astype(np.int64)
    df["timestamp"] = df["timestamp"] - df["timestamp"].min()
    # df["timestamp"] = (
    #     df["timestamp"].apply(lambda x: x.total_seconds()).astype(np.int64)
    # )
    
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Remove continuous outcomes
    df = df[df["correct"].isin([0, 1])]
    df["correct"] = df["correct"].astype(np.int32)

    # Filter nan skills
    if remove_nan_skills:
        df = df[~df["skill_id"].isnull()]
    else:
        df.loc[df["skill_id"].isnull(), "skill_id"] = -1

    user_wise_lst = list()
    for user, user_df in df.groupby("user_id"):
        if len(user_df) >= min_user_inter_num:
            filter_df = user_df.sort_values(by=["timestamp"])  # assure the sequence order
            user_wise_lst.append(filter_df)

    np.random.shuffle(user_wise_lst)
    # user_list = user_wise_lst[:5000]  # sample 5000 students
    user_list = user_wise_lst # all students
    df = pd.concat(user_list).reset_index(drop=True)

    # # Filter too short sequences
    # df = df.groupby("user_id").filter(lambda x: len(x) >= min_user_inter_num)

    df["user_id"] = np.unique(df["user_id"], return_inverse=True)[1]
    df["item_id"] = np.unique(df["item_id"], return_inverse=True)[1]
    df["skill_id"] = np.unique(df["skill_id"].astype(str), return_inverse=True)[1]

    # Build Q-matrix
    Q_mat = np.zeros((len(df["item_id"].unique()), len(df["skill_id"].unique())))
    for item_id, skill_id in df[["item_id", "skill_id"]].values:
        Q_mat[item_id, skill_id] = 1


    print("# Users: {}".format(df["user_id"].nunique()))
    print("# Skills: {}".format(df["skill_id"].nunique()))
    print("# Items: {}".format(df["item_id"].nunique()))
    print("# Interactions: {}".format(len(df)))

    # Get unique skill id from combination of all skill ids
    unique_skill_ids = np.unique(Q_mat, axis=0, return_inverse=True)[1]
    df["skill_id"] = unique_skill_ids[df["item_id"]]

    print("# Preprocessed Skills: {}".format(df["skill_id"].nunique()))
    # Sort data temporally
    df.sort_values(by="timestamp", inplace=True)

    # Sort data by users, preserving temporal order for each user
    df = pd.concat([u_df for _, u_df in df.groupby("user_id")])
    df.to_csv(os.path.join(data_path+"_true", "original_df.csv"), sep="\t", index=False)

    df = df[["user_id", "item_id", "timestamp", "correct", "skill_id"]]

    df.reset_index(inplace=True, drop=True)

    user_num = df["user_id"].nunique()
    skill_num = df["skill_id"].nunique()
    u_skill_all_num = 0
    user_sparsity_5 = 0
    user_sparsity_10 = 0
    user_sparsity_20 = 0
    for _, udf in df.groupby("user_id"):
        u_skill_num = udf["skill_id"].nunique()
        u_skill_all_num += u_skill_num
        user_sparsity_5 += int(u_skill_num <= skill_num * 0.05)
        user_sparsity_10 += int(u_skill_num <= skill_num * 0.1)
        user_sparsity_20 += int(u_skill_num <= skill_num * 0.2)

    print(f'# Sparsity: {((user_num * skill_num - u_skill_all_num) / (user_num * skill_num) * 100):.2f}%')
    print(f'# User_sparsity_ratio_5: {(user_sparsity_5 / user_num * 100):.2f}%')
    print(f'# User_sparsity_ratio_10: {(user_sparsity_10 / user_num * 100):.2f}%')
    print(f'# User_sparsity_ratio_20: {(user_sparsity_20 / user_num * 100):.2f}%')

    # Save data
    with open(os.path.join(data_path+"_true", "question_skill_rel.pkl"), "wb") as f:
        pickle.dump(csr_matrix(Q_mat), f)

    sparse.save_npz(os.path.join(data_path+"_true", "q_mat.npz"), csr_matrix(Q_mat))
    df.to_csv(os.path.join(data_path+"_true", "preprocessed_df.csv"), sep="\t", index=False)



if __name__ == "__main__":
    parser = ArgumentParser(description="Preprocess KT datasets")
    parser.add_argument("--data_name", type=str, default="assistments15")
    parser.add_argument("--min_user_inter_num", type=int, default=5)
    parser.add_argument("--remove_nan_skills", default=True, action="store_true")
    parser.add_argument("--bias", default=False, action="store_true")
    args = parser.parse_args()

    if args.data_name in [
        "assistments09",
        "assistments12",
        "assistments15",
        "assistments17",
    ]:
        prepare_assistments(
            data_name=args.data_name,
            min_user_inter_num=args.min_user_inter_num,
            remove_nan_skills=args.remove_nan_skills,
        )
    elif args.data_name == 'comp':
        prepare_comp(
            data_name=args.data_name,
            min_user_inter_num=args.min_user_inter_num,
            remove_nan_skills=args.remove_nan_skills
        )
    elif args.data_name == "xes3g5m":
        prepare_xes3g5m(
            data_name=args.data_name,
            min_user_inter_num=args.min_user_inter_num,
            remove_nan_skills=args.remove_nan_skills
        )