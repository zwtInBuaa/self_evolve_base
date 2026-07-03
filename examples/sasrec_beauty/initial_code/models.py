#!/usr/bin/env python3
"""
SASRec model for self_evolverec framework.
Core classes from _AAAI_/run_sasrec.py with minimal changes.
"""
from __future__ import annotations

import json
import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ====== SASRec Model Classes (from _AAAI_/run_sasrec.py) ======

class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x.transpose(-1, -2))
        out = F.gelu(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.dropout2(out)
        return out.transpose(-1, -2)


class SASRecBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = PointWiseFeedForward(hidden_dim, dropout)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        residual = x
        q = self.attn_norm(x)
        attn_out, _ = self.attn(q, q, q, attn_mask=attn_mask, need_weights=False)
        x = residual + self.attn_dropout(attn_out)
        x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        residual = x
        y = self.ffn_norm(x)
        y = self.ffn(y)
        x = residual + self.ffn_dropout(y)
        return x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)


class SASRec(nn.Module):
    def __init__(self, num_items: int, max_len: int, hidden_dim: int, num_blocks: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.item_embedding = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.output_embedding = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, hidden_dim)
        self.emb_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([SASRecBlock(hidden_dim, num_heads, dropout) for _ in range(num_blocks)])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.output_embedding.weight)
        nn.init.xavier_uniform_(self.pos_embedding.weight)
        with torch.no_grad():
            self.item_embedding.weight[self.item_embedding.padding_idx].zero_()
            self.output_embedding.weight[self.output_embedding.padding_idx].zero_()

    def output_weight(self) -> torch.Tensor:
        return self.output_embedding.weight[1:]

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(seq.size(1), device=seq.device).unsqueeze(0).expand_as(seq)
        x = self.item_embedding(seq) * math.sqrt(self.hidden_dim)
        x = self.emb_dropout(x + self.pos_embedding(positions))
        key_padding_mask = seq.eq(0)
        attn_mask = torch.triu(torch.ones((seq.size(1), seq.size(1)), device=seq.device, dtype=torch.bool), diagonal=1)
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = self.final_norm(x)
        return x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

    def predict_scores(self, seq: torch.Tensor) -> torch.Tensor:
        hidden = self.forward(seq)
        final_hidden = hidden[:, -1, :]
        return final_hidden @ self.output_weight().t()


def sampled_softmax_loss(hidden: torch.Tensor, target_index: torch.Tensor, output_weight: torch.Tensor, num_negatives: int) -> torch.Tensor:
    if num_negatives <= 0:
        logits = hidden @ output_weight.t()
        logits = torch.clamp(logits, min=-50, max=50)
        return F.cross_entropy(logits, target_index)
    num_classes = output_weight.size(0)
    neg_index = torch.randint(0, num_classes, (target_index.size(0), num_negatives), device=target_index.device)
    target_expanded = target_index.unsqueeze(1)
    collision_mask = neg_index.eq(target_expanded)
    while collision_mask.any():
        neg_index[collision_mask] = torch.randint(0, num_classes, (int(collision_mask.sum().item()),), device=target_index.device)
        collision_mask = neg_index.eq(target_expanded)
    pos_weight = output_weight[target_index]
    neg_weight = output_weight[neg_index]
    pos_logits = (hidden * pos_weight).sum(dim=-1, keepdim=True)
    neg_logits = torch.einsum('bd,bnd->bn', hidden, neg_weight)
    sampled_logits = torch.cat([pos_logits, neg_logits], dim=1)
    sampled_target = torch.zeros(target_index.size(0), dtype=torch.long, device=target_index.device)
    sampled_logits = torch.clamp(sampled_logits, min=-50, max=50)
    return F.cross_entropy(sampled_logits, sampled_target)


# ====== self_evolverec Wrapper ======

class Model(torch.nn.Module):
    """Wrapper for SASRec that matches the self_evolverec interface."""
    def __init__(self, user_num, item_num, eval_args, user_train):
        super(Model, self).__init__()
        self.eval_args = eval_args
        self.user_num = user_num
        self.item_num = item_num
        self.dev = eval_args.device

        self.hidden_dim = 128
        self.num_blocks = 1
        self.num_heads = 1
        self.max_len = 50

        self.sasrec = SASRec(
            num_items=item_num, max_len=self.max_len, hidden_dim=self.hidden_dim,
            num_blocks=self.num_blocks, num_heads=self.num_heads, dropout=0.2,
        )

        with open(f'{eval_args.dataset}/meta.json', 'r', encoding='utf-8') as fs:
            self.meta_data = json.load(fs)
        with open(f'{eval_args.dataset}/train_review.json', 'r', encoding='utf-8') as fs:
            self.review_data = json.load(fs)
        self.user_train = user_train

    def forward(self, user_ids, log_seqs, pos_seqs=None, neg_seqs=None, item_indices=None):
        seqs = torch.LongTensor(log_seqs).to(self.dev)
        if item_indices is not None:
            scores = self.sasrec.predict_scores(seqs)
            candi = torch.LongTensor(item_indices).to(self.dev)
            return torch.gather(scores, 1, candi - 1)
        if pos_seqs is None:
            raise ValueError("Training mode requires pos_seqs.")
        pos_t = torch.LongTensor(pos_seqs).to(self.dev)
        hidden = self.sasrec(seqs)
        mask = pos_t.ne(0)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=self.dev, requires_grad=True)
        return sampled_softmax_loss(hidden[mask], pos_t[mask] - 1, self.sasrec.output_weight(), 500)