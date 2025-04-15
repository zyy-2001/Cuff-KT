import random
import numpy as np
import math
from collections import defaultdict

def counter_kt_seqs(
    q_seq,
    s_seq,
    r_seq,
    t_seq,
    skill_difficulty,
    seed,
    method,
    control,
):

    rng = random.Random(seed)
    np.random.seed(seed)

    counter_q_seq = []
    counter_s_seq = []
    counter_r_seq = []
    pos_q_seq = []
    neg_q_seq = []
    pos_s_seq = []
    neg_s_seq = []
    pos_r_seq = []
    neg_r_seq = []
    logit_r_seq = []
    counter_attention_mask_seq = []
    from collections import defaultdict
    skill_count = defaultdict(int)
    skill_correct = defaultdict(int)
    alpha_double=0.2
    all_alpha = 0
    counter = 0
    for q, s, r in zip(q_seq, s_seq, r_seq):
        prob = rng.random()
        if s!=0:
            contr_val = max(prob, 0.1)*(1-r+(2*r-1)*skill_difficulty[s])
        if s!=0 and contr_val<alpha_double:
            counter_q_seq.append(q)
            counter_s_seq.append(s)
            counter_r_seq.append(1 - r)
            counter_attention_mask_seq.append(1)
        else:
            counter_q_seq.append(q)
            counter_s_seq.append(s)
            counter_r_seq.append(r)
            counter_attention_mask_seq.append(0)
        if s!=0:
            all_alpha += contr_val
            counter+=1
            alpha_double = all_alpha/counter
        skill_count[s] += 1
        skill_correct[s] += r
        logit_r_seq.append(skill_correct[s] / skill_count[s])

    correct_rate_seq, weight_seq = [], []
    correct_count, all_count = 0, 0 
    if method == 'cuff' or method == 'cuff+':
        last_t = defaultdict(lambda: -1)
        count_s = defaultdict(int)
        correct_s = defaultdict(lambda: 1)
        for i, el in enumerate(zip(s_seq, r_seq, t_seq)):
            weight_t = 1
            weight_diff = 1
            s, r, t = el
            if control == 'cuff':
                if r == -1:
                    correct_rate_seq.append(0)
                else:
                    correct_count += r
                    all_count += 1
                    correct_rate_seq.append(correct_count / all_count)
            if r == -1:
                weight_seq.append(1)
                continue
            if i == 0:
                weight_t = 1
            else:
                if last_t[s] == -1:
                    weight_t = 1
                elif abs(t-t_seq[0]) <= 0.01:
                    weight_t = 1
                else:
                    weight_t = (t-last_t[s])/(t-t_seq[0])
            
            if count_s[s] == 0:
                weight_diff = 1
            else:
                weight_diff = abs((correct_s[s] + r)/(count_s[s] + 1) - correct_s[s]/count_s[s]) + 1
            count_s[s] += 1
            correct_s[s] += r
            last_t[s] = t
            weight_seq.append(weight_t * weight_diff)

    return logit_r_seq, counter_attention_mask_seq, correct_rate_seq, weight_seq


def atdkt_kt_seqs(
    s_seq,
    r_seq,
    t_seq,
    method,
    control,
):
    historyratios = []
    right, total = 0, 0
    for i in range(0, len(s_seq)):
        if r_seq[i] == 1:
            right += 1
        total += 1
        historyratios.append(right / total)
    correct_rate_seq, weight_seq = [], []
    correct_count, all_count = 0, 0 
    if method == 'cuff' or method == 'cuff+':
        last_t = defaultdict(lambda: -1)
        count_s = defaultdict(int)
        correct_s = defaultdict(lambda: 1)
        for i, el in enumerate(zip(s_seq, r_seq, t_seq)):
            weight_t = 1
            weight_diff = 1
            s, r, t = el
            if control == 'cuff':
                if r == -1:
                    correct_rate_seq.append(0)
                else:
                    correct_count += r
                    all_count += 1
                    correct_rate_seq.append(correct_count / all_count)
            if r == -1:
                weight_seq.append(1)
                continue
            if i == 0:
                weight_t = 1
            else:
                if last_t[s] == -1:
                    weight_t = 1
                elif abs(t-t_seq[0]) <= 0.01:
                    weight_t = 1
                else:
                    weight_t = (t-last_t[s])/(t-t_seq[0])
            
            if count_s[s] == 0:
                weight_diff = 1
            else:
                weight_diff = abs((correct_s[s] + r)/(count_s[s] + 1) - correct_s[s]/count_s[s]) + 1
            count_s[s] += 1
            correct_s[s] += r
            last_t[s] = t
            weight_seq.append(weight_t * weight_diff)
    return correct_rate_seq, historyratios, weight_seq

def dimkt_kt_seqs(
    q_seq,
    s_seq,
    r_seq,
    t_seq,
    sds,
    qds,
    method,
    control,
):
    
    qd_seq, sd_seq = [], []
    for q, s, r in zip(q_seq, s_seq, r_seq):
        if r != -1:
            qd_seq.append(qds[q-1])
            sd_seq.append(sds[s-1])
        else:
            qd_seq.append(1)
            sd_seq.append(1)

    correct_rate_seq, weight_seq = [], []
    correct_count, all_count = 0, 0 
    if method == 'cuff' or method == 'cuff+':
        last_t = defaultdict(lambda: -1)
        count_s = defaultdict(int)
        correct_s = defaultdict(lambda: 1)
        for i, el in enumerate(zip(s_seq, r_seq, t_seq)):
            weight_t = 1
            weight_diff = 1
            s, r, t = el
            if control == 'cuff':
                if r == -1:
                    correct_rate_seq.append(0)
                else:
                    correct_count += r
                    all_count += 1
                    correct_rate_seq.append(correct_count / all_count)
            if r == -1:
                weight_seq.append(1)
                continue
            if i == 0:
                weight_t = 1
            else:
                if last_t[s] == -1:
                    weight_t = 1
                elif abs(t-t_seq[0]) <= 0.01:
                    weight_t = 1
                else:
                    weight_t = (t-last_t[s])/(t-t_seq[0])
            
            if count_s[s] == 0:
                weight_diff = 1
            else:
                weight_diff = abs((correct_s[s] + r)/(count_s[s] + 1) - correct_s[s]/count_s[s]) + 1
            count_s[s] += 1
            correct_s[s] += r
            last_t[s] = t
            weight_seq.append(weight_t * weight_diff)

    return correct_rate_seq, qd_seq, sd_seq, weight_seq

def cuff_kt_seqs(
    s_seq,
    r_seq,
    t_seq,
    control,
):
    last_t = defaultdict(lambda: -1)
    count_s = defaultdict(int)
    correct_s = defaultdict(lambda: 1)
    correct_rate_seq, weight_seq = [], []
    correct_count, all_count = 0, 0 
    for i, el in enumerate(zip(s_seq, r_seq, t_seq)):
        weight_t = 1
        weight_diff = 1
        s, r, t = el
        if control == 'cuff':
            if r == -1:
                correct_rate_seq.append(0)
            else:
                correct_count += r
                all_count += 1
                correct_rate_seq.append(correct_count / all_count)
        if r == -1:
            weight_seq.append(1)
            continue
        if i == 0:
            weight_t = 1
        else:
            if last_t[s] == -1:
                weight_t = 1
            elif abs(t-t_seq[0]) <= 0.01:
                weight_t = 1
            else:
                weight_t = (t-last_t[s])/(t-t_seq[0])
        
        if count_s[s] == 0:
            weight_diff = 1
        else:
            weight_diff = abs((correct_s[s] + r)/(count_s[s] + 1) - correct_s[s]/count_s[s]) + 1
        count_s[s] += 1
        correct_s[s] += r
        last_t[s] = t
        weight_seq.append(weight_t * weight_diff)
    
    return correct_rate_seq, weight_seq

def preprocess_qr(questions, responses, seq_len, pad_val=-1):
    """
    split the interactions whose length is more than seq_len
    """
    preprocessed_questions = []
    preprocessed_responses = []

    for q, r in zip(questions, responses):
        i = 0
        while i + seq_len < len(q):
            preprocessed_questions.append(q[i : i + seq_len])
            preprocessed_responses.append(r[i : i + seq_len])

            i += seq_len

        preprocessed_questions.append(
            np.concatenate([q[i:], np.array([pad_val] * (i + seq_len - len(q)))])
        )
        preprocessed_responses.append(
            np.concatenate([r[i:], np.array([pad_val] * (i + seq_len - len(q)))])
        )

    return preprocessed_questions, preprocessed_responses


def preprocess_qsr(questions, skills, responses, seq_len, pad_val=0):
    """
    split the interactions whose length is more than seq_len
    """
    preprocessed_questions = []
    preprocessed_skills = []
    preprocessed_responses = []
    attention_mask = []

    for q, s, r in zip(questions, skills, responses):
        i = 0
        while i + seq_len < len(q):
            preprocessed_questions.append(q[i : i + seq_len])
            preprocessed_skills.append(s[i : i + seq_len])
            preprocessed_responses.append(r[i : i + seq_len])
            attention_mask.append(np.ones(seq_len))
            i += seq_len

        preprocessed_questions.append(
            np.concatenate([q[i:], np.array([pad_val] * (i + seq_len - len(q)))])
        )
        preprocessed_skills.append(
            np.concatenate([s[i:], np.array([pad_val] * (i + seq_len - len(q)))])
        )
        preprocessed_responses.append(
            np.concatenate([r[i:], np.array([-1] * (i + seq_len - len(q)))])
        )
        attention_mask.append(
            np.concatenate(
                [np.ones_like(r[i:]), np.array([0] * (i + seq_len - len(q)))]
            )
        )

    return (
        preprocessed_questions,
        preprocessed_skills,
        preprocessed_responses,
        attention_mask,
    )
