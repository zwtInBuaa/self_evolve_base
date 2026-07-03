#!/usr/bin/env python3
"""
Random recommender main_code for self_evolverec framework.
Core logic from _AAAI_/run_random.py, data from data_cache/Pet/.
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models import Model, RandomRecommender


# ====== Data Loading (from data_cache JSON format) ======

def load_sequences_from_json(dataset_path: str) -> Tuple[Dict[int, List[int]], int, int]:
    with open(f'{dataset_path}/test.json', 'r', encoding='utf-8') as f:
        test = json.load(f)
    user_sequences: Dict[int, List[int]] = {}
    max_item = 0
    for uid_str, seq in test['History'].items():
        uid = int(uid_str)
        if len(seq) < 3:
            continue
        user_sequences[uid] = seq
        max_item = max(max_item, max(seq))
    max_user = max(user_sequences.keys()) if user_sequences else 0
    return user_sequences, max_user, max_item


@dataclass
class UserSplit:
    train: List[int]
    val: int
    test: int
    full: set[int]


def build_leave_one_out_splits(user_sequences: Dict[int, List[int]]) -> Dict[int, UserSplit]:
    splits: Dict[int, UserSplit] = {}
    for user, seq in user_sequences.items():
        if len(seq) < 3:
            continue
        splits[user] = UserSplit(train=seq[:-2], val=seq[-2], test=seq[-1], full=set(seq))
    return splits


def build_eval_sequences(splits: Dict[int, UserSplit], users: Sequence[int], max_len: int, mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
    seqs, targets = [], []
    for user in users:
        split = splits[user]
        if mode == 'val':
            history, target = split.train, split.val
        elif mode == 'test':
            history, target = split.train + [split.val], split.test
        else:
            raise ValueError(f'Unknown eval mode: {mode}')
        arr = np.zeros(max_len, dtype=np.int64)
        history = history[-max_len:]
        arr[max_len - len(history):] = np.array(history, dtype=np.int64)
        seqs.append(arr)
        targets.append(target)
    return torch.from_numpy(np.stack(seqs)), torch.tensor(targets, dtype=torch.long)


def evaluate_exact(model: RandomRecommender, splits: Dict[int, UserSplit], users: Sequence[int],
                   max_len: int, eval_batch_size: int, device: torch.device, mode: str) -> Dict[str, float]:
    model.eval()
    hit5 = hit10 = ndcg5 = ndcg10 = 0.0
    total = 0
    with torch.inference_mode():
        for start in range(0, len(users), eval_batch_size):
            batch_users = users[start:start + eval_batch_size]
            seqs, targets = build_eval_sequences(splits, batch_users, max_len=max_len, mode=mode)
            seqs = seqs.to(device)
            targets = targets.to(device)
            scores = model.predict_scores(seqs)
            for row_idx, user in enumerate(batch_users):
                history = splits[user].train if mode == 'val' else (splits[user].train + [splits[user].val])
                target = int(targets[row_idx].item())
                mask_items = [item for item in history if int(item) != target]
                if mask_items:
                    hist_tensor = torch.tensor(mask_items, device=device, dtype=torch.long) - 1
                    scores[row_idx, hist_tensor] = -1e9
            _, top10 = torch.topk(scores, k=10, dim=1)
            top10 = top10 + 1
            top5 = top10[:, :5]
            for row_idx in range(len(batch_users)):
                target = int(targets[row_idx].item())
                pred5 = top5[row_idx].tolist()
                pred10 = top10[row_idx].tolist()
                if target in pred5:
                    hit5 += 1.0
                    ndcg5 += 1.0 / math.log2(pred5.index(target) + 2.0)
                if target in pred10:
                    hit10 += 1.0
                    ndcg10 += 1.0 / math.log2(pred10.index(target) + 2.0)
                total += 1
    if total == 0:
        return {'Recall@5': 0.0, 'Recall@10': 0.0, 'NDCG@5': 0.0, 'NDCG@10': 0.0}
    return {'Recall@5': hit5 / total, 'Recall@10': hit10 / total, 'NDCG@5': ndcg5 / total, 'NDCG@10': ndcg10 / total}


# ====== Main entry point for self_evolverec ======

def main(args):
    logger = logging.getLogger(__name__)

    # ---- Config (from _AAAI_/run_random.py) ----
    dataset_path = "data_cache/Pet"
    max_len = 50
    batch_size = 128
    eval_batch_size = 128
    epochs = 20
    strategy = 'popularity'
    pop_temperature = 1.0
    seed = 20260521
    early_stop_patience = 3
    selection_metric = 'Recall@10'
    device = torch.device(args.device)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ---- Load data ----
    logger.info(f"Loading data from {dataset_path}...")
    user_sequences, num_users, num_items = load_sequences_from_json(dataset_path)
    splits = build_leave_one_out_splits(user_sequences)
    eval_users = sorted([u for u, s in splits.items() if len(s.train) >= 2])
    logger.info(f"Data loaded: {num_users} users, {num_items} items, {len(eval_users)} eval_users")

    # ---- Build model ----
    model = RandomRecommender(num_items=num_items, strategy=strategy, pop_temperature=pop_temperature, seed=seed).to(device)
    model.fit_popularity(splits)

    # ---- "Train" (just evaluate multiple times) ----
    best_val = -1.0
    best_test = None
    no_improve_evals = 0
    training_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        val_metrics = evaluate_exact(model, splits, eval_users, max_len, eval_batch_size, device, 'val')
        val_score = val_metrics[selection_metric]

        if val_score > best_val:
            best_val = val_score
            best_val_metrics = {k: float(v) for k, v in val_metrics.items()}
            best_test = evaluate_exact(model, splits, eval_users, max_len, eval_batch_size, device, 'test')
            no_improve_evals = 0
        else:
            no_improve_evals += 1
            if early_stop_patience > 0 and no_improve_evals >= early_stop_patience:
                break

        logger.info(f"Epoch {epoch}/{epochs}: val_R@5={val_metrics['Recall@5']:.4f}, "
                    f"val_R@10={val_score:.4f}, val_N@5={val_metrics['NDCG@5']:.4f}, "
                    f"time={time.time()-epoch_start:.0f}s, best={best_val:.4f}, no_impr={no_improve_evals}")

    if best_test is None:
        best_test = evaluate_exact(model, splits, eval_users, max_len, eval_batch_size, device, 'test')

    logger.info(f"Test: R@5={best_test['Recall@5']:.4f}, R@10={best_test['Recall@10']:.4f}, "
                f"N@5={best_test['NDCG@5']:.4f}, N@10={best_test['NDCG@10']:.4f}")

    return 1, {
        "ndcg_score(0-1)": best_test['NDCG@5'],
        "hr_score(0-1)": best_test['Recall@5'],
        "combined_score": (0.6 * best_test['Recall@5'] + 0.4 * best_test['NDCG@5']),
        "simulator_comment": f"Random baseline: R@10={best_test['Recall@10']:.4f}, N@10={best_test['NDCG@10']:.4f}",
        "diagnosis_comment": f"Non-learned baseline. Strategy={strategy}, pop_temp={pop_temperature}",
    }


if __name__ == '__main__':
    from simulator.util_functions import get_args
    args = get_args()
    status, results = main(args)
    print(f"Status: {status}")
    print(f"Results: {results}")