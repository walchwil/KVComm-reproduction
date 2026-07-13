import importlib.util
import os
import re
from pathlib import Path

from .base_evaluator import BaseEvaluator


LATENTMAS_ZDS_DIR = Path(
    os.environ.get(
        "LATENTMAS_ZDS_DIR",
        "/home/user2/zhangdongsen/Latent_comunication/LatentMAS-zds",
    )
)

DATA_PY = LATENTMAS_ZDS_DIR / "data.py"


def _load_latentmas_data_module():
    """
    Load LatentMAS-zds/data.py while resolving its local `from utils import ...`
    against LatentMAS-zds/utils.py instead of KVComm/utils.
    """
    if not DATA_PY.exists():
        raise FileNotFoundError(f"Cannot find LatentMAS-zds data.py at {DATA_PY}")

    import sys

    zds_dir = str(LATENTMAS_ZDS_DIR)

    old_sys_path = list(sys.path)
    old_utils = sys.modules.get("utils", None)
    had_utils = "utils" in sys.modules

    try:
        # Put LatentMAS-zds first so `import utils` resolves to its local utils.py.
        sys.path.insert(0, zds_dir)

        # If KVComm/utils has already been imported as top-level `utils`, remove it
        # temporarily; otherwise Python will reuse the cached wrong module.
        if had_utils:
            del sys.modules["utils"]

        spec = importlib.util.spec_from_file_location("latentmas_zds_data", str(DATA_PY))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    finally:
        sys.path = old_sys_path

        # Restore previous top-level utils module if it existed.
        if had_utils:
            sys.modules["utils"] = old_utils
        else:
            sys.modules.pop("utils", None)


def _split_arc_question(question: str):
    lines = [x.strip() for x in str(question).splitlines() if x.strip()]
    if not lines:
        return "", ""
    stem = lines[0]
    choices = "\n".join(lines[1:])
    return stem, choices


def _normalize_gold(item):
    if item.get("gold") is not None:
        return str(item["gold"]).strip()
    if item.get("solution") is not None:
        return str(item["solution"]).strip()
    return ""


def _to_kvcomm_item(task: str, item: dict):
    """
    Convert LatentMAS-zds samples into KVComm prompt_A/prompt_B format.

    Convention in KVComm:
      prompt_A = sender / Ms context
      prompt_B = receiver / Mr query + answer-format instruction
    """
    question = str(item.get("question", "")).strip()
    solution = str(item.get("solution", "")).strip()
    gold = _normalize_gold(item)

    if task in {"arc_easy", "arc_challenge"}:
        stem, choices = _split_arc_question(question)

        # ARC has no explicit context/question split.
        # We use a choice-context split:
        #   sender/Ms: candidate options
        #   receiver/Mr: question stem + output-format instruction
        prompt_A = choices
        prompt_B = (
            f"{stem}\n\n"
            "Return only the option letter, such as a, b, c, or d."
        )
        answer = gold

    elif task in {"aime2024", "aime2025"}:
        # AIME has no separate context. Treat the full problem as sender context;
        # receiver only receives the instruction to solve using communicated context.
        prompt_A = question
        prompt_B = (
            "Use the sender context to solve the math problem. "
            "Return only the final integer answer, without explanation."
        )
        answer = gold

    elif task == "gpqa":
        # GPQA loader gives a full question string and gold answer text.
        # Since we do not have structured choices here, put the full question in sender context.
        prompt_A = question
        prompt_B = (
            "Use the sender context to answer the question. "
            "Return only the final answer, without explanation."
        )
        answer = solution if solution else gold

    elif task == "mbppplus":
        # MBPP+ question contains the programming problem and example tests.
        # Treat these as sender-side problem specification.
        prompt_A = question
        prompt_B = (
            "Use the sender context to write a correct self-contained Python solution. "
            "Return only the Python code in a markdown code block."
        )
        answer = solution if solution else gold

    elif task == "humanevalplus":
        # HumanEval+ question contains the function signature/docstring-style task.
        # Treat it as sender-side problem specification.
        prompt_A = question
        prompt_B = (
            "Use the sender context to complete the programming problem with a correct Python implementation. "
            "Return only the Python code in a markdown code block."
        )
        answer = solution if solution else gold

    else:
        raise ValueError(f"Unsupported latentmas task for KVComm: {task}")

    return {
        "idx": item.get("idx", 0),
        "prompt_A": prompt_A,
        "prompt_B": prompt_B,
        "answer": answer,
    }


class LatentMASTaskEvaluator(BaseEvaluator):
    def __init__(self, task: str):
        super().__init__()
        self.task = task
        self.name = task
        self.max_tokens = 256 if task in {"mbppplus", "humanevalplus"} else 96
        self.truncate_input = True
        self.multiple_answers = False
        self.n_samples = None
        self.data = self.load_data()

    def load_data(self):
        mod = _load_latentmas_data_module()

        if self.task == "aime2024":
            loader = mod.load_aime2024
            split = "train"
        elif self.task == "aime2025":
            loader = mod.load_aime2025
            split = "train"
        elif self.task == "gpqa":
            loader = mod.load_gpqa_diamond
            split = "test"
        elif self.task == "arc_easy":
            loader = mod.load_arc_easy
            split = "test"
        elif self.task == "arc_challenge":
            loader = mod.load_arc_challenge
            split = "test"
        elif self.task == "mbppplus":
            loader = mod.load_mbppplus
            split = "test"
        elif self.task == "humanevalplus":
            loader = mod.load_humanevalplus
            split = "test"
        else:
            raise ValueError(f"Unsupported task: {self.task}")

        rows = []
        for item in loader(split=split):
            rows.append(_to_kvcomm_item(self.task, dict(item)))

        if not rows:
            raise RuntimeError(f"No samples loaded for task={self.task}")

        print(f"[LatentMAS task] Loaded {len(rows)} rows for {self.task} split={split}", flush=True)
        return rows
