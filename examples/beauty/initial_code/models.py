import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
from tqdm import tqdm


class RelativePositionalBias(nn.Module):
    def __init__(self, max_seq_len: int) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.weight = nn.Parameter(torch.empty(2 * max_seq_len - 1).normal_(mean=0.0, std=0.02))

    def forward(self, seq_len: int) -> torch.Tensor:
        positions = torch.arange(seq_len, device=self.weight.device)
        rel = positions[:, None] - positions[None, :]
        rel = rel + self.max_seq_len - 1
        return self.weight[rel]


class HSTUBlock(nn.Module):
    def __init__(
        self, hidden_dim, linear_dim, attention_dim, num_heads, max_len, dropout, attn_dropout,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear_dim = linear_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.norm_input = nn.LayerNorm(hidden_dim)
        self.uvqk = nn.Linear(hidden_dim, num_heads * (2 * linear_dim + 2 * attention_dim), bias=False)
        self.rel_bias = RelativePositionalBias(max_len)
        self.norm_attn = nn.LayerNorm(num_heads * linear_dim)
        self.output = nn.Linear(num_heads * linear_dim, hidden_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.uvqk.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, padding_mask):
        batch_size, seq_len, _ = x.shape
        normed_x = self.norm_input(x)
        uvqk = self.uvqk(normed_x)
        split_sizes = [
            self.num_heads * self.linear_dim,
            self.num_heads * self.linear_dim,
            self.num_heads * self.attention_dim,
            self.num_heads * self.attention_dim,
        ]
        u, v, q, k = torch.split(uvqk, split_sizes, dim=-1)
        u, v, q, k = F.silu(u), F.silu(v), F.silu(q), F.silu(k)

        u = u.view(batch_size, seq_len, self.num_heads, self.linear_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.linear_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.attention_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.attention_dim)

        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool))
        valid_key_mask = ~padding_mask
        attn_mask = causal_mask.unsqueeze(0) & valid_key_mask.unsqueeze(1)

        attn = torch.einsum('bnhd,bmhd->bhnm', q, k)
        attn = F.silu(attn + self.rel_bias(seq_len).unsqueeze(0).unsqueeze(0))
        attn = attn / max(seq_len, 1)
        attn = attn.masked_fill(~attn_mask.unsqueeze(1), 0.0)
        attn = self.attn_dropout(attn)

        attn_output = torch.einsum('bhnm,bmhd->bnhd', attn, v).reshape(batch_size, seq_len, self.num_heads * self.linear_dim)
        gated = u.reshape(batch_size, seq_len, self.num_heads * self.linear_dim) * self.norm_attn(attn_output)
        out = self.output(self.dropout(gated)) + x
        return out.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class HSTUEncoder(nn.Module):
    """HSTU encoder: SiLU-gated attention, L2-normalized output, sampled softmax."""
    def __init__(
        self, num_items, max_len, hidden_dim, linear_dim, attention_dim,
        num_blocks, num_heads, dropout, attn_dropout,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.item_embedding = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, hidden_dim)
        self.emb_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            HSTUBlock(hidden_dim, linear_dim, attention_dim, num_heads, max_len, dropout, attn_dropout)
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.item_embedding.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
        nn.init.xavier_normal_(self.pos_embedding.weight)
        with torch.no_grad():
            self.item_embedding.weight[self.item_embedding.padding_idx].zero_()

    def output_weight(self):
        return F.normalize(self.item_embedding.weight[1:], p=2, dim=-1, eps=1e-6)

    def forward(self, seq):
        positions = torch.arange(seq.size(1), device=seq.device).unsqueeze(0).expand_as(seq)
        x = self.item_embedding(seq) * math.sqrt(self.hidden_dim)
        x = x + self.pos_embedding(positions)
        padding_mask = seq.eq(0)
        x = self.emb_dropout(x).masked_fill(padding_mask.unsqueeze(-1), 0.0)
        for block in self.blocks:
            x = block(x, padding_mask=padding_mask)
        x = self.final_norm(x)
        return x.masked_fill(padding_mask.unsqueeze(-1), 0.0), padding_mask


class Model(torch.nn.Module):
    """Wrapper for HSTU that matches the self_evolverec interface."""
    def __init__(self, user_num, item_num, eval_args, user_train):
        super(Model, self).__init__()
        self.eval_args = eval_args
        self.user_num = user_num
        self.item_num = item_num
        self.dev = eval_args.device

        # HSTU hyperparameters
        self.hidden_dim = getattr(eval_args, 'hidden_units', 128)
        self.linear_dim = getattr(eval_args, 'linear_dim', 16)
        self.attention_dim = getattr(eval_args, 'attention_dim', 16)
        self.num_blocks = getattr(eval_args, 'num_blocks', 1)
        self.num_heads = getattr(eval_args, 'num_heads', 1)
        self.temperature = getattr(eval_args, 'temperature', 0.05)
        self.num_negatives = getattr(eval_args, 'num_negatives', 500)
        self.max_len = eval_args.maxlen

        self.encoder = HSTUEncoder(
            num_items=item_num,
            max_len=self.max_len,
            hidden_dim=self.hidden_dim,
            linear_dim=self.linear_dim,
            attention_dim=self.attention_dim,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            dropout=eval_args.dropout_rate,
            attn_dropout=getattr(eval_args, 'attn_dropout', 0.0),
        )

        # Load meta and review data
        with open(f'{eval_args.dataset}/meta.json', 'r', encoding='utf-8') as fs:
            self.meta_data = json.load(fs)
        with open(f'{eval_args.dataset}/train_review.json', 'r', encoding='utf-8') as fs:
            self.review_data = json.load(fs)
        self.user_train = user_train

    def log2feats(self, log_seqs):
        """Encode input sequences to hidden states (backward compat)."""
        seqs = torch.LongTensor(log_seqs).to(self.dev)
        hidden, _ = self.encoder(seqs)
        return hidden

    def predict_score(self, user_seq_emb, item_indices):
        """Score items given user representation.
        item_indices: (batch, seq_len) or (batch, seq_len, num_candidates), 1-indexed
        """
        user_vec = F.normalize(user_seq_emb[:, -1, :], p=2, dim=-1, eps=1e-6)  # (B, D)
        output_w = self.encoder.output_weight()  # (num_items, D)

        if item_indices.dim() == 2:
            # (batch, seq_len) -> score each position
            item_embs = output_w[item_indices - 1]  # (B, S, D)
            logits = (user_vec.unsqueeze(1) * item_embs).sum(dim=-1)  # (B, S)
        elif item_indices.dim() == 3:
            # (batch, seq_len, num_candidates) -> score each candidate
            B, S, C = item_indices.shape
            flat_idx = (item_indices - 1).reshape(B, S * C)
            item_embs = output_w[flat_idx]  # (B, S*C, D)
            logits = (user_vec.unsqueeze(1) * item_embs).sum(dim=-1).reshape(B, S, C)  # (B, S, C)
            logits = logits / self.temperature
        else:
            raise ValueError(f"Unexpected item_indices shape: {item_indices.shape}")
        return logits

    def _sampled_softmax_loss(self, hidden, target_index):
        """Sampled softmax cross-entropy loss."""
        temperature = max(float(self.temperature), 1e-8)
        output_weight = self.encoder.output_weight()
        num_classes = output_weight.size(0)
        num_neg = self.num_negatives

        if num_neg <= 0:
            logits = (hidden @ output_weight.t()) / temperature
            return F.cross_entropy(logits, target_index)

        neg_index = torch.randint(0, num_classes, (target_index.size(0), num_neg), device=target_index.device)
        target_expanded = target_index.unsqueeze(1)
        collision_mask = neg_index.eq(target_expanded)
        while collision_mask.any():
            neg_index[collision_mask] = torch.randint(0, num_classes, (int(collision_mask.sum().item()),), device=target_index.device)
            collision_mask = neg_index.eq(target_expanded)

        pos_weight = output_weight[target_index]
        neg_weight = output_weight[neg_index]
        pos_logits = (hidden * pos_weight).sum(dim=-1, keepdim=True) / temperature
        neg_logits = torch.einsum('bd,bnd->bn', hidden, neg_weight) / temperature
        sampled_logits = torch.cat([pos_logits, neg_logits], dim=1)
        sampled_target = torch.zeros(target_index.size(0), dtype=torch.long, device=target_index.device)
        return F.cross_entropy(sampled_logits, sampled_target)

    def forward(self, user_ids, log_seqs, pos_seqs=None, neg_seqs=None, item_indices=None):
        """
        HSTU forward:
        - Training: use sampled softmax loss on pos_seqs (neg_seqs is ignored)
        - Inference: score item_indices
        """
        seqs = torch.LongTensor(log_seqs).to(self.dev)
        hidden, _ = self.encoder(seqs)

        if item_indices is not None:
            user_vec = F.normalize(hidden[:, -1, :], p=2, dim=-1, eps=1e-6)
            output_w = self.encoder.output_weight()
            item_embs = output_w[torch.LongTensor(item_indices).to(self.dev) - 1]
            logits = (user_vec.unsqueeze(1) * item_embs).sum(dim=-1)
            return logits.squeeze(-1)

        if pos_seqs is None:
            raise ValueError("Training mode requires pos_seqs.")

        # Sampled softmax loss using all positive positions
        pos_t = torch.LongTensor(pos_seqs).to(self.dev)
        mask = pos_t.ne(0)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=self.dev, requires_grad=True)

        hidden_valid = F.normalize(hidden[mask], p=2, dim=-1, eps=1e-6)
        loss = self._sampled_softmax_loss(hidden_valid, pos_t[mask] - 1)
        return loss