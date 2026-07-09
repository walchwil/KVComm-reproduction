from .base_evaluator import BaseEvaluator
from .medqa_utils import build_medqa_prompts
from datasets import load_dataset
import os
import sys


class MedQAEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        self.max_tokens = 64
        self.truncate_input = True
        self.multiple_answers = False
        self.n_samples = None
        self.data = self.load_data()
        self.medqa = True
        self.name = "medqa"

    def load_data(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = [
            os.path.join(project_root, "data", "medqa.json"),
            os.path.join(os.path.dirname(project_root), "medqa.json"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "medqa.json"),
        ]

        dataset_path = None
        for path in candidates:
            if os.path.exists(path):
                dataset_path = path
                break

        if dataset_path is None:
            raise FileNotFoundError(
                "Cannot find medqa.json. Expected one of:\n" + "\n".join(candidates)
            )

        print(f"[MedQA] Loading {dataset_path}", file=sys.stderr, flush=True)
        dataset = load_dataset("json", data_files=dataset_path, split="train")
        print(f"[MedQA] Loaded {len(dataset)} rows; cleaning prompts...", file=sys.stderr, flush=True)
        dataset = dataset.map(build_medqa_prompts, desc="Cleaning MedQA")
        keep_columns = {"prompt_A", "prompt_B", "answer"}
        remove_columns = [name for name in dataset.column_names if name not in keep_columns]
        if remove_columns:
            dataset = dataset.remove_columns(remove_columns)
        print("[MedQA] Cleaning complete", file=sys.stderr, flush=True)
        return dataset
