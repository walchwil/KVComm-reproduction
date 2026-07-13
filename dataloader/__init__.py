from .latentmas_tasks import LatentMASTaskEvaluator
def get_evaluator(test_task: str):
    if test_task in {"arc_easy", "arc_challenge", "humanevalplus", "mbppplus", "gpqa", "aime2024", "aime2025"}:
        return LatentMASTaskEvaluator(test_task)
    if test_task == "countries":
        from .countries import CountriesEvaluator
        return CountriesEvaluator()
    elif test_task == "tipsheets":
        from .tipsheets import TipsheetsEvaluator
        return TipsheetsEvaluator()
    elif test_task == "hotpotqa":
        from .hotpotqa import HotpotQAEvaluator
        return HotpotQAEvaluator()
    elif test_task == "hotpotqa_full":
        from .hotpotqa import HotpotQAEvaluator
        return HotpotQAEvaluator(n_samples=None)
    elif test_task == "qasper":
        from .qasper import QaSperEvaluator
        return QaSperEvaluator()
    elif test_task == "qasper_full":
        from .qasper import QaSperEvaluator
        return QaSperEvaluator(n_samples=None)
    elif test_task == "musique":
        from .musique import MuSiQueEvaluator
        return MuSiQueEvaluator()
    elif test_task == "musique_full":
        from .musique import MuSiQueEvaluator
        return MuSiQueEvaluator(n_samples=None)
    elif test_task == "multifieldqa_en":
        from .multifieldqa_en import MultiFieldQAEnEvaluator
        return MultiFieldQAEnEvaluator()
    elif test_task == "twowikimqa":
        from .twowikimqa import TwoWikiMQAEvaluator
        return TwoWikiMQAEvaluator()
    elif test_task == "tmath":
        from .tmath import TMathEvaluator
        return TMathEvaluator()
    elif test_task == "repobench":
        from .repobench import RepoBenchEvaluator
        return RepoBenchEvaluator()
    elif test_task == "samsum":
        from .samsum import SAMSumEvaluator
        return SAMSumEvaluator()
    elif test_task == "medqa":
        from .medqa import MedQAEvaluator
        return MedQAEvaluator()
    else:
        raise ValueError(f"Unsupported task name: {test_task}")

def get_multi_agent_evaluator(test_task: str):
    if test_task == "hotpotqa":
        from .hotpotqa import HotpotQAEvaluator
        return HotpotQAEvaluator(multi_agent=True)
    elif test_task == "musique":
        from .musique import MuSiQueEvaluator
        return MuSiQueEvaluator(multi_agent=True)
    elif test_task == "twowikimqa":
        from .twowikimqa import TwoWikiMQAEvaluator
        return TwoWikiMQAEvaluator(multi_agent=True)
    else:
        raise ValueError(f"Unsupported task name: {test_task}")

def get_mix_evaluator(test_task: str, mix_method: str):
    if test_task == "countries_tipsheets":
        from .countries_tipsheets import CountriesTipsheetsEvaluator
        return CountriesTipsheetsEvaluator(mix_method=mix_method)
    elif test_task == "countries_multifieldqa":
        from .countries_multifieldqa_en import CountriesMultiFieldQAEvaluator
        return CountriesMultiFieldQAEvaluator(mix_method=mix_method)
    else:
        raise ValueError(f"Unsupported task name: {test_task}")
