"""Minimal diagnosis tools for Random recommender (no learned parameters)."""
import json


def diagnosis_interpreter_prompt(raw_diagnosis):
    return "Random recommender: no learned parameters to diagnose."


class DiagnosisProbe:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.meta_data = model.meta_data
        self.review_data = model.review_data
        self.user_train = model.user_train

    def run_full_diagnosis(self, train_loader):
        return {
            "metrics": {"embedding_collapse_score": 0.0},
            "metric_definitions": {"embedding_collapse_score": "Random recommender has no learned embeddings."},
        }
