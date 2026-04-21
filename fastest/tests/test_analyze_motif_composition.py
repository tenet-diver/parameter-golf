import unittest
from types import SimpleNamespace

from fastest.scripts import analyze_motif_composition


class AnalyzeMotifCompositionTests(unittest.TestCase):
    def test_pair_rows_include_interaction_labels_and_recommendations_split_try_vs_avoid(self) -> None:
        records = [
            SimpleNamespace(tags=["a", "b"], val_bpb=1.19, path="records/a_b_1.json"),
            SimpleNamespace(tags=["a", "b"], val_bpb=1.20, path="records/a_b_2.json"),
            SimpleNamespace(tags=["a", "c"], val_bpb=1.21, path="records/a_c_1.json"),
            SimpleNamespace(tags=["a", "c"], val_bpb=1.22, path="records/a_c_2.json"),
            SimpleNamespace(tags=["b", "c"], val_bpb=1.2002, path="records/b_c_1.json"),
            SimpleNamespace(tags=["b", "c"], val_bpb=1.1998, path="records/b_c_2.json"),
        ]
        single_summary = {
            "a": {"mean_val_bpb": 1.2},
            "b": {"mean_val_bpb": 1.2},
            "c": {"mean_val_bpb": 1.2},
        }

        pair_rows = analyze_motif_composition.build_pair_rows(records, single_summary)
        labels_by_pair = {tuple(row["pair"]): row["interaction"] for row in pair_rows}

        self.assertEqual("positive", labels_by_pair[("a", "b")])
        self.assertEqual("negative", labels_by_pair[("a", "c")])
        self.assertEqual("neutral", labels_by_pair[("b", "c")])

        recommendations = analyze_motif_composition.build_recommendations(pair_rows)

        self.assertIn("combinations_to_try", recommendations)
        self.assertIn("combinations_to_avoid", recommendations)
        self.assertTrue(recommendations["combinations_to_try"])
        self.assertTrue(recommendations["combinations_to_avoid"])


if __name__ == "__main__":
    unittest.main()
