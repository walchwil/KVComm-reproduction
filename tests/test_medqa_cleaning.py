import unittest
import importlib.util
from pathlib import Path

HELPER_PATH = Path(__file__).resolve().parents[1] / "dataloader" / "medqa_utils.py"
spec = importlib.util.spec_from_file_location("medqa_utils", HELPER_PATH)
medqa_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(medqa_utils)

format_options = medqa_utils.format_options
split_before_options = medqa_utils.split_before_options
split_case_and_leadin = medqa_utils.split_case_and_leadin


class MedQACleaningTests(unittest.TestCase):
    def test_splits_case_from_leadin_and_removes_duplicate_options(self):
        options = [
            "A. Hemolytic uremic syndrome",
            "B. Oral ulcers",
            "C. Colorectal cancer",
            "D. Pancreatic cancer",
        ]
        raw = (
            "A 34-year-old man comes to the physician because of a 3-week history "
            "of colicky abdominal pain and diarrhea. Colonoscopy shows a bleeding, "
            "ulcerated rectal mucosa with several pseudopolyps. Which of the "
            "following is this patient at greatest risk of developing?\n"
            "A. Hemolytic uremic syndrome\n"
            "B. Oral ulcers\n"
            "C. Colorectal cancer\n"
            "D. Pancreatic cancer\n"
            "Choose the correct option."
        )

        stem = split_before_options(raw, options)
        case_context, leadin = split_case_and_leadin(stem)

        self.assertIn("Colonoscopy shows", case_context)
        self.assertNotIn("A. Hemolytic", case_context)
        self.assertNotIn("Choose the correct option", case_context)
        self.assertEqual(
            leadin,
            "Which of the following is this patient at greatest risk of developing?",
        )

    def test_formats_unlabeled_options_with_letters(self):
        options = ["alpha", "beta"]

        self.assertEqual(format_options(options), "A. alpha\nB. beta")


if __name__ == "__main__":
    unittest.main()
