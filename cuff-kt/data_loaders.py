from torch.utils.data import Dataset
import os
from utils.augment_seq import (
    preprocess_qr,
    preprocess_qsr,
    atdkt_kt_seqs,
    dimkt_kt_seqs,
    cuff_kt_seqs,
    counter_kt_seqs,
)
import torch
from collections import defaultdict

class CounterDatasetWrapper(Dataset):
    def __init__(
        self,
        ds: Dataset,
        seq_len: int,
        method: str,
        control,
    ):
        super().__init__()
        self.ds = ds
        self.seq_len = seq_len
        self.method = method
        self.control = control

        self.num_questions = self.ds.num_questions
        self.num_skills = self.ds.num_skills
        self.q_mask_id = self.num_questions + 1
        self.s_mask_id = self.num_skills + 1
        self.skill_difficulty = self.ds.skill_difficulty

    def __len__(self):
        return len(self.ds)

    def __getitem_internal__(self, index):
        original_data = self.ds[index]
        q_seq = original_data["questions"]
        s_seq = original_data["skills"]
        r_seq = original_data["responses"]
        attention_mask = original_data["attention_mask"]
        q_seq_list = original_data["questions"].tolist()
        s_seq_list = original_data["skills"].tolist()
        r_seq_list = original_data["responses"].tolist()
        t_seq_list = original_data["times"].tolist()

        t = counter_kt_seqs(
            q_seq_list,
            s_seq_list,
            r_seq_list,
            t_seq_list,
            self.skill_difficulty,
            seed=index,
            method=self.method,
            control=self.control,
        )

        logit_r_seq, counter_attention_mask, correct_rates, weight_seq = t

        logit_r_seq = torch.tensor(logit_r_seq, dtype=torch.float)
        counter_attention_mask = torch.tensor(counter_attention_mask, dtype=torch.long)
        attention_reweight = torch.tensor(weight_seq, dtype=torch.float)
        correct_rates = torch.tensor(correct_rates, dtype=torch.float)

        ret = {
            "questions": q_seq,
            "skills": s_seq,
            "responses": (r_seq, logit_r_seq),
            "attention_mask": (counter_attention_mask, attention_mask),
            "attention_reweight": attention_reweight,
            "correct_rates": correct_rates,
        }
        return ret

    def __getitem__(self, index):
        return self.__getitem_internal__(index)

class ATDKTDatasetWrapper(Dataset):
    def __init__(
        self,
        ds: Dataset,
        seq_len: int,
        method: str,
        control,
    ):
        super().__init__()
        self.ds = ds
        self.seq_len = seq_len
        self.method = method
        self.control = control

        self.num_questions = self.ds.num_questions
        self.num_skills = self.ds.num_skills
        self.q_mask_id = self.num_questions + 1
        self.s_mask_id = self.num_skills + 1
        self.skill_difficulty = self.ds.skill_difficulty

    def __len__(self):
        return len(self.ds)

    def __getitem_internal__(self, index):
        original_data = self.ds[index]
        q_seq = original_data["questions"]
        s_seq = original_data["skills"]
        r_seq = original_data["responses"]
        t_seq = original_data["times"]
        attention_mask = original_data["attention_mask"]
        q_seq_list = original_data["questions"].tolist()
        s_seq_list = original_data["skills"].tolist()
        r_seq_list = original_data["responses"].tolist()
        t_seq_list = original_data["times"].tolist()

        t = atdkt_kt_seqs(
            s_seq_list,
            r_seq_list,
            t_seq_list,
            self.method,
            self.control,
        )

        correct_rates, historyratios, weight_seq = t

        historyratios = torch.tensor(historyratios, dtype=torch.float)
        attention_reweight = torch.tensor(weight_seq, dtype=torch.float)
        correct_rates = torch.tensor(correct_rates, dtype=torch.float)

        ret = {
            "questions": q_seq,
            "skills": s_seq,
            "responses": r_seq,
            "historycorrs": historyratios,
            "attention_mask": attention_mask,
            "attention_reweight": attention_reweight,
            "correct_rates": correct_rates,
        }
        return ret

    def __getitem__(self, index):
        return self.__getitem_internal__(index)
    

class DimKTDatasetWrapper(Dataset):
    def __init__(
        self,
        ds: Dataset,
        seq_len: int,
        sds,
        qds,
        method,
        control,
    ):
        super().__init__()
        self.ds = ds
        self.seq_len = seq_len
        self.sds = sds
        self.qds = qds
        self.method = method
        self.control = control

        self.num_questions = self.ds.num_questions
        self.num_skills = self.ds.num_skills
        self.q_mask_id = self.num_questions + 1
        self.s_mask_id = self.num_skills + 1
        self.skill_difficulty = self.ds.skill_difficulty

    def __len__(self):
        return len(self.ds)

    def __getitem_internal__(self, index):
        original_data = self.ds[index]
        q_seq = original_data["questions"]
        s_seq = original_data["skills"]
        r_seq = original_data["responses"]
        t_seq = original_data["times"]
        attention_mask = original_data["attention_mask"]
        q_seq_list = original_data["questions"].tolist()
        s_seq_list = original_data["skills"].tolist()
        r_seq_list = original_data["responses"].tolist()
        t_seq_list = original_data["times"].tolist()

        t = dimkt_kt_seqs(
            q_seq_list,
            s_seq_list,
            r_seq_list,
            t_seq_list,
            self.sds,
            self.qds,
            self.method,
            self.control,
        )

        correct_rates, question_difficulty, skill_difficulty, weight_seq = t

        question_difficulty = torch.tensor(question_difficulty, dtype=torch.long)
        skill_difficulty = torch.tensor(skill_difficulty, dtype=torch.long)
        attention_reweight = torch.tensor(weight_seq, dtype=torch.float)
        correct_rates = torch.tensor(correct_rates, dtype=torch.float)

        ret = {
            "questions": q_seq,
            "skills": s_seq,
            "responses": r_seq,
            "question_difficulty": question_difficulty,
            "skill_difficulty": skill_difficulty,
            "attention_mask": attention_mask,
            "attention_reweight": attention_reweight,
            "correct_rates": correct_rates,
        }
        return ret

    def __getitem__(self, index):
        return self.__getitem_internal__(index)

class CuffDatasetWrapper(Dataset):
    def __init__(
        self,
        ds: Dataset,
        seq_len: int,
        control,
    ):
        super().__init__()
        self.ds = ds
        self.seq_len = seq_len
        self.control = control

        self.num_questions = self.ds.num_questions
        self.num_skills = self.ds.num_skills
        self.q_mask_id = self.num_questions + 1
        self.s_mask_id = self.num_skills + 1
        self.skill_difficulty = self.ds.skill_difficulty

    def __len__(self):
        return len(self.ds)

    def __getitem_internal__(self, index):
        original_data = self.ds[index]
        q_seq = original_data["questions"]
        s_seq = original_data["skills"]
        r_seq = original_data["responses"]
        t_seq = original_data["times"]
        attention_mask = original_data["attention_mask"]
        s_seq_list = original_data["skills"].tolist()
        r_seq_list = original_data["responses"].tolist()
        t_seq_list = original_data["times"].tolist()

        t = cuff_kt_seqs(
            s_seq_list,
            r_seq_list,
            t_seq_list,
            self.control,
        )

        correct_rates, weight_seq = t

        correct_rates = torch.tensor(correct_rates, dtype=torch.float)
        attention_reweight = torch.tensor(weight_seq, dtype=torch.float)

        ret = {
            "questions": q_seq,
            "skills": s_seq,
            "responses": r_seq,
            "attention_mask": attention_mask,
            "attention_reweight": attention_reweight,
            "correct_rates": correct_rates,
        }
        return ret

    def __getitem__(self, index):
        return self.__getitem_internal__(index)

class MostRecentQuestionSkillDataset(Dataset):
    def __init__(self, df, seq_len, num_skills, num_questions):
        self.df = df
        self.seq_len = seq_len
        self.num_skills = num_skills
        self.num_questions = num_questions

        self.questions = [
            u_df["item_id"].values[-self.seq_len :]
            for _, u_df in self.df.groupby("user_id")
        ]
        self.skills = [
            u_df["skill_id"].values[-self.seq_len :]
            for _, u_df in self.df.groupby("user_id")
        ]
        self.responses = [
            u_df["correct"].values[-self.seq_len :]
            for _, u_df in self.df.groupby("user_id")
        ]
        self.times = [
            u_df["timestamp"].values[-self.seq_len :]
            for _, u_df in self.df.groupby("user_id")
        ]
        self.lengths = [
            len(u_df["skill_id"].values) for _, u_df in self.df.groupby("user_id")
        ]

        skill_correct = defaultdict(int)
        skill_count = defaultdict(int)
        for s_list, r_list in zip(self.skills, self.responses):
            for s, r in zip(s_list, r_list):
                skill_correct[s] += r
                skill_count[s] += 1

        self.skill_difficulty = {
            s: skill_correct[s] / float(skill_count[s]) for s in skill_correct
        }

        # print(f'diff = {self.skill_difficulty}')
        # import sys
        # sys.exit()


        cnt = 0
        for interactions in self.questions:
            cnt += len(interactions)
        self.num_interactions = cnt

        self.len = len(self.questions)

        self.padded_q = torch.zeros(
            (len(self.questions), self.seq_len), dtype=torch.long
        )
        self.padded_s = torch.zeros((len(self.skills), self.seq_len), dtype=torch.long)
        self.padded_t = torch.zeros((len(self.times), self.seq_len), dtype=torch.long)
        self.padded_r = torch.full(
            (len(self.responses), self.seq_len), -1, dtype=torch.long
        )
        self.attention_mask = torch.zeros(
            (len(self.skills), self.seq_len), dtype=torch.long
        )
        # self.attention_reweight = torch.ones(
        #     (len(self.skills), self.seq_len), dtype=torch.float
        # )

        for i, elem in enumerate(zip(self.questions, self.skills, self.responses, self.times)):
            q, s, r, t = elem
            
            self.padded_q[i, -len(q) :] = torch.tensor(q, dtype=torch.long)
            self.padded_s[i, -len(s) :] = torch.tensor(s, dtype=torch.long)
            self.padded_r[i, -len(r) :] = torch.tensor(r, dtype=torch.long)
            self.padded_t[i, -len(t) :] = torch.tensor(t, dtype=torch.long)
            self.attention_mask[i, -len(s) :] = torch.ones(len(s), dtype=torch.long)

    def __getitem__(self, index):

        return {
            "questions": self.padded_q[index],
            "skills": self.padded_s[index],
            "responses": self.padded_r[index],
            "times": self.padded_t[index],
            "attention_mask": self.attention_mask[index],
        }

    def __len__(self):
        return self.len
    
class PreprocessQuestionSkillDataset(Dataset):
    def __init__(self, df, seq_len, num_skills, num_questions):
        self.df = df
        self.seq_len = seq_len
        self.num_skills = num_skills
        self.num_questions = num_questions

        self.user_ids = self.df["user_id"].unique().tolist()

        self.questions, self.skills, self.responses, self.times = [], [], [], []

        for user_id in self.user_ids:
            user_df = self.df[self.df["user_id"] == user_id]
            self.questions.append(user_df["item_id"].values[-self.seq_len:])
            self.skills.append(user_df["skill_id"].values[-self.seq_len:])
            self.responses.append(user_df["correct"].values[-self.seq_len:])
            self.times.append(user_df["timestamp"].values[-self.seq_len:])


        self.correct_rate_list = []
        skill_correct = defaultdict(int)
        skill_count = defaultdict(int)
        for s_list, r_list in zip(self.skills, self.responses):
            correct_count, all_count = 0, 0
            for s, r in zip(s_list, r_list):
                skill_correct[s] += r
                skill_count[s] += 1
                correct_count += r
                all_count += 1
            self.correct_rate_list.append(correct_count / all_count)

        self.skill_difficulty = {
            s: skill_correct[s] / float(skill_count[s]) for s in skill_correct
        }


        self.len = len(self.questions)

        self.padded_q = torch.zeros(
            (len(self.questions), self.seq_len), dtype=torch.long
        )
        self.padded_s = torch.zeros((len(self.skills), self.seq_len), dtype=torch.long)
        self.padded_t = torch.zeros((len(self.times), self.seq_len), dtype=torch.long)
        self.padded_r = torch.full(
            (len(self.responses), self.seq_len), -1, dtype=torch.long
        )
        self.attention_mask = torch.zeros(
            (len(self.skills), self.seq_len), dtype=torch.long
        )

        for i, elem in enumerate(zip(self.questions, self.skills, self.responses, self.times)):
            q, s, r, t = elem
            
            self.padded_q[i, -len(q) :] = torch.tensor(q, dtype=torch.long)
            self.padded_s[i, -len(s) :] = torch.tensor(s, dtype=torch.long)
            self.padded_r[i, -len(r) :] = torch.tensor(r, dtype=torch.long)
            self.padded_t[i, -len(t) :] = torch.tensor(t, dtype=torch.long)
            self.attention_mask[i, -len(s) :] = torch.ones(len(s), dtype=torch.long)

    def __getitem__(self, index):

        return {
            "questions": self.padded_q[index],
            "skills": self.padded_s[index],
            "responses": self.padded_r[index],
            "times": self.padded_t[index],
            "attention_mask": self.attention_mask[index],
            "stu_ids": self.user_ids[index],
            "correct_rates": self.correct_rate_list[index],
        }

    def __len__(self):
        return self.len

class SelectQuestionSkillDataset(Dataset):
    def split_list(self, input_list, chunk_size=100, min_remainder=5):
        result = []
        for row in input_list:
            row_result = [row[i:i + chunk_size] for i in range(0, len(row), chunk_size)]
            if len(row_result) == 0:
                continue
            if len(row_result[-1]) <= min_remainder:
                row_result.pop()  

            result.extend(row_result)
        return result

    def __init__(self, df, seq_len, num_skills, num_questions, ini, lst):
        '''
        ini: the ratio of the first position
        lst: the ratio of the last position
        '''
        self.df = df
        self.seq_len = seq_len
        self.num_skills = num_skills
        self.num_questions = num_questions

        self.questions = [
            u_df["item_id"].values[int(len(u_df["item_id"])*ini): int(len(u_df["item_id"])*lst)]
            for _, u_df in self.df.groupby("user_id")
        ]
        self.skills = [
            u_df["skill_id"].values[int(len(u_df["skill_id"])*ini): int(len(u_df["skill_id"])*lst)]
            for _, u_df in self.df.groupby("user_id")
        ]
        self.responses = [
            u_df["correct"].values[int(len(u_df["correct"])*ini): int(len(u_df["correct"])*lst)]
            for _, u_df in self.df.groupby("user_id")
        ]
        self.times = [
            u_df["timestamp"].values[int(len(u_df["timestamp"])*ini): int(len(u_df["timestamp"])*lst)]
            for _, u_df in self.df.groupby("user_id")
        ]


        self.questions = self.split_list(self.questions)
        self.skills = self.split_list(self.skills)
        self.responses = self.split_list(self.responses)
        self.times = self.split_list(self.times)


        skill_correct = defaultdict(int)
        skill_count = defaultdict(int)
        for s_list, r_list in zip(self.skills, self.responses):
            for s, r in zip(s_list, r_list):
                skill_correct[s] += r
                skill_count[s] += 1

        self.skill_difficulty = {
            s: skill_correct[s] / float(skill_count[s]) for s in skill_correct
        }

        cnt = 0
        for interactions in self.questions:
            cnt += len(interactions)
        self.num_interactions = cnt

        self.len = len(self.questions)

        self.padded_q = torch.zeros(
            (len(self.questions), self.seq_len), dtype=torch.long
        )
        self.padded_s = torch.zeros((len(self.skills), self.seq_len), dtype=torch.long)
        self.padded_t = torch.zeros((len(self.times), self.seq_len), dtype=torch.long)
        self.padded_r = torch.full(
            (len(self.responses), self.seq_len), -1, dtype=torch.long
        )
        self.attention_mask = torch.zeros(
            (len(self.skills), self.seq_len), dtype=torch.long
        )
        # self.attention_reweight = torch.ones(
        #     (len(self.skills), self.seq_len), dtype=torch.float
        # )

        for i, elem in enumerate(zip(self.questions, self.skills, self.responses, self.times)):
            q, s, r, t = elem
            
            self.padded_q[i, -len(q) :] = torch.tensor(q, dtype=torch.long)
            self.padded_s[i, -len(s) :] = torch.tensor(s, dtype=torch.long)
            self.padded_r[i, -len(r) :] = torch.tensor(r, dtype=torch.long)
            self.padded_t[i, -len(t) :] = torch.tensor(t, dtype=torch.long)
            self.attention_mask[i, -len(s) :] = torch.ones(len(s), dtype=torch.long)

    def __getitem__(self, index):

        return {
            "questions": self.padded_q[index],
            "skills": self.padded_s[index],
            "responses": self.padded_r[index],
            "times": self.padded_t[index],
            "attention_mask": self.attention_mask[index],
        }

    def __len__(self):
        return self.len


class MostEarlyQuestionSkillDataset(Dataset):
    def __init__(self, df, seq_len, num_skills, num_questions):
        self.df = df
        self.seq_len = seq_len
        self.num_skills = num_skills
        self.num_questions = num_questions

        self.questions = [
            u_df["item_id"].values[: self.seq_len]
            for _, u_df in self.df.groupby("user_id")
        ]
        self.skills = [
            u_df["skill_id"].values[: self.seq_len]
            for _, u_df in self.df.groupby("user_id")
        ]
        self.times = [
            u_df["timestamp"].values[-self.seq_len :]
            for _, u_df in self.df.groupby("user_id")
        ]
        self.responses = [
            u_df["correct"].values[: self.seq_len]
            for _, u_df in self.df.groupby("user_id")
        ]
        self.lengths = [
            len(u_df["skill_id"].values) for _, u_df in self.df.groupby("user_id")
        ]

        cnt = 0
        for interactions in self.questions:
            cnt += len(interactions)
        self.num_interactions = cnt

        self.len = len(self.questions)

        self.padded_q = torch.zeros(
            (len(self.questions), self.seq_len), dtype=torch.long
        )
        self.padded_s = torch.zeros((len(self.skills), self.seq_len), dtype=torch.long)
        self.padded_t = torch.zeros((len(self.times), self.seq_len), dtype=torch.long)
        self.padded_r = torch.full(
            (len(self.responses), self.seq_len), -1, dtype=torch.long
        )
        self.attention_mask = torch.zeros(
            (len(self.skills), self.seq_len), dtype=torch.long
        )

        for i, elem in enumerate(zip(self.questions, self.skills, self.responses, self.times)):
            q, s, r, t = elem
            self.padded_q[i, : len(q)] = torch.tensor(q, dtype=torch.long)
            self.padded_s[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            self.padded_r[i, : len(r)] = torch.tensor(r, dtype=torch.long)
            self.padded_t[i, -len(t) :] = torch.tensor(t, dtype=torch.long)
            self.attention_mask[i, : len(r)] = torch.ones(len(s), dtype=torch.long)

    def __getitem__(self, index):
        return {
            "questions": self.padded_q[index],
            "skills": self.padded_s[index],
            "responses": self.padded_r[index],
            "times": self.padded_t[index],
            "attention_mask": self.attention_mask[index],
        }

    def __len__(self):
        return self.len


class SkillDataset(Dataset):
    def __init__(self, df, seq_len, num_skills, num_questions):
        self.df = df
        self.seq_len = seq_len
        self.num_skills = num_skills
        self.num_questions = num_questions

        self.questions = [
            u_df["skill_id"].values for _, u_df in self.df.groupby("user_id")
        ]
        self.responses = [
            u_df["correct"].values for _, u_df in self.df.groupby("user_id")
        ]
        self.lengths = [
            len(u_df["skill_id"].values) for _, u_df in self.df.groupby("user_id")
        ]

        cnt = 0
        for interactions in self.questions:
            cnt += len(interactions)
        self.num_interactions = cnt

        self.questions, self.responses = preprocess_qr(
            self.questions, self.responses, self.seq_len
        )
        self.len = len(self.questions)

    def __getitem__(self, index):
        return {"questions": self.questions[index], "responses": self.responses[index]}

    def __len__(self):
        return self.len
