#!/usr/bin/env python3
"""
HSTU main_code for self_evolverec framework.
Training logic adapted from _AAAI_/run_hstu.py, data loaded from data_cache/Pet/.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models import Model, HSTU, sampled_softmax_loss


# ====== Data Loading (adapted to data_cache JSON format) ======

def load_sequences_from_json(dataset_path: str) -> Tuple[Dict[int, List[int]], int, int]:
    """Load sequences from data_cache JSON format into same structure as sequential_data.txt."""
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


class HSTUTrainDataset(Dataset):
    def __init__(
        self,
        splits: Dict[int, UserSplit],
        max_len: int,
        train_window_size: int,
        train_targets: str,
    ) -> None:
        self.users = [user for user, split in splits.items() if len(split.train) >= 2]
        self.splits = splits
        self.max_len = max_len
        self.train_window_size = train_window_size
        self.train_targets = train_targets

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, index: int):
        user = self.users[index]
        train = self.splits[user].train
        effective_window = self.max_len if self.train_window_size <= 0 else min(self.max_len, self.train_window_size)
        seq = np.zeros(self.max_len, dtype=np.int64)
        if self.train_targets == 'last_position':
            history = train[:-1][-effective_window:]
            target = train[-1]
            seq[self.max_len - len(history):] = np.array(history, dtype=np.int64)
            return torch.from_numpy(seq), torch.tensor(target, dtype=torch.long)
        pos = np.zeros(self.max_len, dtype=np.int64)
        history = train[-(effective_window + 1):]
        src = history[:-1]
        tgt = history[1:]
        seq[self.max_len - len(src):] = np.array(src, dtype=np.int64)
        pos[self.max_len - len(tgt):] = np.array(tgt, dtype=np.int64)
        return torch.from_numpy(seq), torch.from_numpy(pos)


def build_eval_sequences(
    splits: Dict[int, UserSplit],
    users: Sequence[int],
    max_len: int,
    mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    seqs = []
    targets = []
    for user in users:
        split = splits[user]
        if mode == 'val':
            history = split.train
            target = split.val
        elif mode == 'test':
            history = split.train + [split.val]
            target = split.test
        else:
            raise ValueError(f'Unknown eval mode: {mode}')
        arr = np.zeros(max_len, dtype=np.int64)
        history = history[-max_len:]
        arr[max_len - len(history):] = np.array(history, dtype=np.int64)
        seqs.append(arr)
        targets.append(target)
    return torch.from_numpy(np.stack(seqs)), torch.tensor(targets, dtype=torch.long)


def evaluate_exact(
    model: HSTU,
    splits: Dict[int, UserSplit],
    users: Sequence[int],
    max_len: int,
    eval_batch_size: int,
    device: torch.device,
    mode: str,
) -> Dict[str, float]:
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
    return {
        'Recall@5': hit5 / total,
        'Recall@10': hit10 / total,
        'NDCG@5': ndcg5 / total,
        'NDCG@10': ndcg10 / total,
    }


# ====== Main entry point for self_evolverec ======

def main(args):
    logger = logging.getLogger(__name__)

    # ---- Config ----
    dataset_path = "data_cache/Pet"
    # HSTU params from _AAAI_/run_hstu.py
    max_len = 50
    hidden_dim = 128
    linear_dim = 16
    attention_dim = 16
    num_blocks = 1
    num_heads = 1
    dropout = 0.5
    attn_dropout = 0.0
    batch_size = 128
    eval_batch_size = 128
    epochs = 20
    lr = 3e-4
    num_negatives = 500
    temperature = 0.05
    train_targets = 'all_positions'
    train_window_size = 0
    early_stop_patience = 3
    seed = 20260521
    device = torch.device(args.device)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ---- Load data ----
    logger.info(f"Loading data from {dataset_path}...")
    user_sequences, num_users, num_items = load_sequences_from_json(dataset_path)
    splits = build_leave_one_out_splits(user_sequences)
    train_users = [user for user, split in splits.items() if len(split.train) >= 2]
    eval_users = sorted(train_users)
    logger.info(f"Data loaded: {num_users} users, {num_items} items, {len(train_users)} train_users, {len(eval_users)} eval_users")

    dataset = HSTUTrainDataset(splits=splits, max_len=max_len, train_window_size=train_window_size, train_targets=train_targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=False)

    # ---- Build model ----
    model = HSTU(
        num_items=num_items, max_len=max_len, hidden_dim=hidden_dim, linear_dim=linear_dim,
        attention_dim=attention_dim, num_blocks=num_blocks, num_heads=num_heads,
        dropout=dropout, attn_dropout=attn_dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-8)

    # ---- Train ----
    best_val = -1.0
    best_state = None
    no_improve_evals = 0
    max_epoch_seconds = 30 * 60

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        epoch_loss = 0.0
        step_count = 0

        for step, batch in enumerate(loader, start=1):
            if train_targets == 'last_position':
                seq, target = [x.to(device) for x in batch]
                hidden = F.normalize(model(seq)[:, -1, :], p=2, dim=-1, eps=1e-6)
                loss = sampled_softmax_loss(hidden, target - 1, model.output_weight(), int(num_negatives), float(temperature))
            else:
                seq, pos = [x.to(device) for x in batch]
                hidden = model(seq)
                mask = pos.ne(0)
                if mask.sum() == 0:
                    continue
                hidden_valid = F.normalize(hidden[mask], p=2, dim=-1, eps=1e-6)
                loss = sampled_softmax_loss(hidden_valid, pos[mask] - 1, model.output_weight(), int(num_negatives), float(temperature))

            if torch.isnan(loss):
                return -1, "Nan"

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += float(loss.item())
            step_count += 1

            # Time check
            if step > 5:
                elapsed = time.time() - epoch_start
                if elapsed > max_epoch_seconds:
                    return 1, {
                        "ndcg_score(0-1)": 0,
                        "hr_score(0-1)": 0,
                        "combined_score": 0.0,
                        "simulator_comment": f"Training too slow: {elapsed:.0f}s per epoch",
                        "diagnosis_comment": "",
                    }

        # ---- Validate ----
        val_metrics = evaluate_exact(model=model, splits=splits, users=eval_users, max_len=max_len,
                                     eval_batch_size=eval_batch_size, device=device, mode='val')
        val_score = val_metrics['Recall@10']

        epoch_sec = time.time() - epoch_start
        logger.info(f"Epoch {epoch}/{epochs}: loss={epoch_loss/max(step_count,1):.4f}, "
                    f"val_R@5={val_metrics['Recall@5']:.4f}, val_R@10={val_score:.4f}, "
                    f"val_N@5={val_metrics['NDCG@5']:.4f}, time={epoch_sec:.0f}s, "
                    f"best={best_val:.4f}, no_impr={no_improve_evals}")

        if val_score > best_val:
            best_val = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val_metrics = {k: float(v) for k, v in val_metrics.items()}
            no_improve_evals = 0
        else:
            no_improve_evals += 1
            if early_stop_patience > 0 and no_improve_evals >= early_stop_patience:
                break

    # ---- Test ----
    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_exact(model=model, splits=splits, users=eval_users, max_len=max_len,
                                  eval_batch_size=eval_batch_size, device=device, mode='test')
    logger.info(f"Test: R@5={test_metrics['Recall@5']:.4f}, R@10={test_metrics['Recall@10']:.4f}, "
                f"N@5={test_metrics['NDCG@5']:.4f}, N@10={test_metrics['NDCG@10']:.4f}")

    best_valid_hr = best_val_metrics.get('Recall@5', 0.0) if best_state else 0.0
    best_valid_ndcg = best_val_metrics.get('NDCG@5', 0.0) if best_state else 0.0

    return 1, {
        "ndcg_score(0-1)": best_valid_ndcg,
        "hr_score(0-1)": best_valid_hr,
        "combined_score": (0.6 * best_valid_hr + 0.4 * best_valid_ndcg),
        "simulator_comment": f"HSTU Beauty: Recall@10={test_metrics['Recall@10']:.4f}, NDCG@10={test_metrics['NDCG@10']:.4f}",
        "diagnosis_comment": f"Best val: Recall@5={best_valid_hr:.4f}, NDCG@5={best_valid_ndcg:.4f}",
    }


if __name__ == '__main__':
    from simulator.util_functions import get_args
    args = get_args()
    status, results = main(args)
    print(f"Status: {status}")
    print(f"Results: {results}")