import re


QUESTION_START_PATTERNS = [
    r"Which of the following",
    r"Which one of the following",
    r"What is the most likely",
    r"What is the best",
    r"What is the next",
    r"What is the appropriate",
    r"What is the diagnosis",
    r"What is the cause",
    r"What is this patient",
    r"What should",
    r"Which of these",
]


def normalize_space(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def format_options(options) -> str:
    if not options:
        return ""

    lines = []
    for i, opt in enumerate(options):
        opt = normalize_space(str(opt))
        if re.match(r"^[A-Z]\.\s+", opt):
            lines.append(opt)
        else:
            label = chr(ord("A") + i)
            lines.append(f"{label}. {opt}")
    return "\n".join(lines)


def strip_choose_instruction(text: str) -> str:
    text = normalize_space(text)
    text = re.sub(r"\n?Choose the correct option\.?\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?Choose the single best answer\.?\s*$", "", text, flags=re.IGNORECASE)
    return normalize_space(text)


def split_before_options(text: str, options) -> str:
    text = strip_choose_instruction(text)

    if options:
        first_option = normalize_space(str(options[0]))
        pos = text.find(first_option)
        if pos != -1:
            return normalize_space(text[:pos])

        match = re.search(r"\nA\.\s+", text)
        if match:
            return normalize_space(text[:match.start()])

    return text


def split_case_and_leadin(stem_with_leadin: str) -> tuple[str, str]:
    text = normalize_space(stem_with_leadin)

    best_start = -1
    for pattern in QUESTION_START_PATTERNS:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        if matches:
            best_start = max(best_start, matches[-1].start())

    if best_start != -1:
        return normalize_space(text[:best_start]), normalize_space(text[best_start:])

    if "?" in text:
        qmark = text.rfind("?")
        before_q = text[: qmark + 1]
        cut = max(before_q.rfind(". "), before_q.rfind("\n"))
        if cut != -1:
            return normalize_space(before_q[: cut + 1]), normalize_space(before_q[cut + 1 :])

    return text, "Choose the correct option based on the clinical context."


def build_medqa_prompts(sample: dict) -> dict:
    options = sample.get("options", [])
    answer = normalize_space(sample.get("answer", ""))
    raw = sample.get("query") or sample.get("question") or ""

    stem_with_leadin = split_before_options(raw, options)
    case_context, leadin_question = split_case_and_leadin(stem_with_leadin)
    options_text = format_options(options)

    prompt_B = (
        "Question:\n"
        f"{leadin_question}\n\n"
        "Options:\n"
        f"{options_text}\n\n"
        "Choose the single best answer. Return the exact answer text only, without explanation."
    )

    return {
        "prompt_A": case_context,
        "prompt_B": prompt_B,
        "answer": answer,
    }
