"""Minimal diagnosis tools for HSTU model."""
import torch
import numpy as np
import torch.nn.functional as F
import json
from collections import Counter


def diagnosis_interpreter_prompt(raw_diagnosis):
    return f"""
    You are a Senior AI Researcher analyzing the mathematical health of a Recommendation System.

    Here are the raw mathematical metrics measured from the current model:
    {json.dumps(raw_diagnosis['metrics'], indent=2)}

    Here are the definitions of what these metrics mean:
    {json.dumps(raw_diagnosis['metric_definitions'], indent=2)}

    Your task: produce a concise diagnosis summary.
    Strictly follow the JSON format below.

    Output Format (JSON):
    {{
    "status": <CRITICAL | NEEDS_IMPROVEMENT | STABLE>,
    "core_findings": ["<1-2 sentences: key interpretation>"],
    "key_implications": ["<1 sentence: what this implies>"],
    "evidence": {{
        "headline_metrics": {{"<metric_name>": <value>}},
        "brief_metric_read": {{"<metric_name>": "<short interpretation>"}}
    }}
    }}
    """


class DiagnosisProbe:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.meta_data = model.meta_data
        self.review_data = model.review_data
        self.user_train = model.user_train

    def analyze_embeddings_collapse_cosine(self, sample_size: int = 2048, seed: int = 42):
        with torch.no_grad():
            emb = self.model.hstu.item_embedding.weight.detach()
            emb = emb[1:]
            n = emb.size(0)
            if n <= 2:
                return 0.0
            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)
            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            x = emb.index_select(0, idx)
            x = F.normalize(x, p=2, dim=-1)
            gram = x @ x.t()
            offdiag_sum = gram.sum() - gram.diag().sum()
            offdiag_cnt = k * (k - 1)
            mean_cos = (offdiag_sum / offdiag_cnt).clamp(-1, 1).item()
            collapse_score = (mean_cos + 1.0) * 0.5
            return round(float(collapse_score), 4)

    def run_full_diagnosis(self, train_loader):
        collapse_score = self.analyze_embeddings_collapse_cosine(sample_size=2048)
        return {
            "metrics": {
                "embedding_collapse_score": collapse_score,
            },
            "metric_definitions": {
                "embedding_collapse_score": (
                    "Range ~[0,1]. Estimated by mean pairwise cosine similarity of sampled item embeddings "
                    "(mapped to [0,1]). Higher means embeddings point in similar directions (worse diversity / more collapse)."
                ),
            },
        }