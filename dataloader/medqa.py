from .base_evaluator import BaseEvaluator
from .medqa_utils import build_medqa_prompts
from datasets import load_dataset
import os
import sys
import re


def _strip_option_prefix(text):
    text = str(text).strip()
    return re.sub(r"^[A-Da-d][\.\):：\s]+", "", text).strip()


def normalize_medqa_response(response, options=None):
    """
    Normalize MedQA model outputs such as:
      C
      C.
      C. Colorectal cancer
      Answer: C
      The answer is C. Colorectal cancer
    into the corresponding option text when possible.
    """
    response = str(response).strip()
    if not response:
        return response

    # Only use the first non-empty line for strict answer extraction.
    lines = [x.strip() for x in response.splitlines() if x.strip()]
    first = lines[0] if lines else response

    # Remove common answer prefixes.
    first = re.sub(
        r"^(final\s+answer|answer|the\s+answer\s+is)\s*[:：]?\s*",
        "",
        first,
        flags=re.I,
    ).strip()

    # Try to catch a leading option letter: C, C., (C), C:
    m = re.match(r"^\(?([A-Da-d])\)?[\.\):：\s]*(.*)$", first)
    if m:
        letter = m.group(1).upper()
        rest = m.group(2).strip()
        idx = ord(letter) - ord("A")

        # Prefer mapping from options if options are available.
        if isinstance(options, dict):
            # Most MedQA json files use {"A": "...", "B": "..."}.
            if letter in options:
                return _strip_option_prefix(options[letter])
            if letter.lower() in options:
                return _strip_option_prefix(options[letter.lower()])

        if isinstance(options, list) and 0 <= idx < len(options):
            return _strip_option_prefix(options[idx])

        # Fallback: if response is "C. Colorectal cancer", return the text part.
        if rest:
            return _strip_option_prefix(rest)

        # Last fallback: just return the letter.
        return letter

    # If the full response contains an option text, normalize to that option.
    option_values = []
    if isinstance(options, dict):
        option_values = list(options.values())
    elif isinstance(options, list):
        option_values = options

    lower_resp = response.lower()
    for opt in option_values:
        clean_opt = _strip_option_prefix(opt)
        if clean_opt and clean_opt.lower() in lower_resp:
            return clean_opt

    return response
import sys


class MedQAEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        self.max_tokens = 96
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
        print(f"[MedQA] Loading {dataset_path}", file=sys.stderr, flush=True)
        dataset = load_dataset("json", data_files=dataset_path, split="train")
        print(f"[MedQA] Loaded {len(dataset)} rows; cleaning prompts...", file=sys.stderr, flush=True)

        dataset = dataset.map(build_medqa_prompts, desc="Cleaning MedQA")

        # Keep options so that evaluate_item can map "A/B/C/D" to answer text.
        keep_columns = {"prompt_A", "prompt_B", "answer", "options"}
        remove_columns = [name for name in dataset.column_names if name not in keep_columns]
        if remove_columns:
            dataset = dataset.remove_columns(remove_columns)

        print("[MedQA] Cleaning complete", file=sys.stderr, flush=True)
        return dataset

    def evaluate_item(self, item, response):
        pred = normalize_medqa_response(response, item.get("options", None))
        return super().evaluate_item(item, pred)
