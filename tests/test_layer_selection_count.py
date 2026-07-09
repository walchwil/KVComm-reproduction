import unittest

from layer_importance import get_num_selected_layers


class LayerSelectionCountTests(unittest.TestCase):
    def test_uses_ceil_for_fractional_layer_counts(self):
        self.assertEqual(get_num_selected_layers(0.3, 36), 11)

    def test_selects_at_least_one_layer_for_positive_ratio(self):
        self.assertEqual(get_num_selected_layers(0.01, 36), 1)


if __name__ == "__main__":
    unittest.main()
