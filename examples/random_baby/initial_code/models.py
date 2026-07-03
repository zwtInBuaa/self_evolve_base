#!/usr/bin/env python3
"""
Random recommender model for self_evolverec framework.
Core classes from _AAAI_/run_random.py with minimal changes.
"""
from __future__ import annotations

import json
from typing import Dict, List

import torch
import torch.nn as nn


class RandomRecommender(nn.Module):
    """Non-learned baseline. Scores are random; 'popularity' multiplies the
    uniform random draw by item-frequency^(1/temperature)."""

    def __init__(self, num_items: int, strategy: str, pop_temperature: float, seed: int) -> None:
        super().__init__()
        self.num_items = num_items
        self.strategy = strategy
        self.pop_temperature = max(float(pop_temperature), 1e-6)
        self.register_buffer('item_pop', torch.zeros(num_items))
        self.rng = torch.Generator(device='cpu').manual_seed(int(seed))

    def fit_popularity(self, splits: Dict[int, any]) -> None:
        pop = torch.zeros(self.num_items)
        for split in splits.values():
            for it in split.train:
                if 1 <= it <= self.num_items:
                    pop[it - 1] += 1.0
        self.item_pop.copy_(pop)

    def predict_scores(self, seq: torch.Tensor) -> torch.Tensor:
        batch_size = seq.size(0)
        scores = torch.rand(batch_size, self.num_items, generator=self.rng)
        if self.strategy == 'popularity':
            weights = self.item_pop.clamp(min=1.0) ** (1.0 / self.pop_temperature)
            scores = scores * weights.unsqueeze(0)
        return scores


class Model(torch.nn.Module):
    """Wrapper for RandomRecommender that matches the self_evolverec interface."""
    def __init__(self, user_num, item_num, eval_args, user_train):
        super(Model, self).__init__()
        self.eval_args = eval_args
        self.user_num = user_num
        self.item_num = item_num
        self.dev = eval_args.device

        self.random = RandomRecommender(
            num_items=item_num,
            strategy='popularity',
            pop_temperature=1.0,
            seed=20260521,
        )

        with open(f'{eval_args.dataset}/meta.json', 'r', encoding='utf-8') as fs:
            self.meta_data = json.load(fs)
        with open(f'{eval_args.dataset}/train_review.json', 'r', encoding='utf-8') as fs:
            self.review_data = json.load(fs)
        self.user_train = user_train

    def forward(self, user_ids, log_seqs, pos_seqs=None, neg_seqs=None, item_indices=None):
        return torch.tensor(0.0, device=self.dev, requires_grad=True)