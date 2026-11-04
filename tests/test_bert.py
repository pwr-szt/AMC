import unittest

from amc.wykop import bert

PREDICTED = {
    "job_insecurity": "no",
    "insecurity_cause": "economy",
    "sector": "it",
    "stance": "reassurance",
    "attitude": "negative",
    "threat_type": "job_loss",
    "competence_claim": "none",
}


class GateTests(unittest.TestCase):
    def test_rules_follow_stage1_ai_and_earlier_labels(self):
        labels = bert.gate("yes", PREDICTED)
        self.assertEqual(labels["insecurity_cause"], "none")  # no insecurity
        self.assertEqual(labels["threat_type"], "none")  # stance is not a threat
        self.assertEqual(labels["attitude"], "negative")

    def test_text_without_ai_gets_no_ai_labels(self):
        labels = bert.gate("no", {**PREDICTED, "job_insecurity": "yes", "stance": "threat"})
        self.assertEqual(labels["insecurity_cause"], "economy")
        for label in ("stance", "attitude", "threat_type", "competence_claim"):
            self.assertEqual(labels[label], "none")

    def test_threat_keeps_threat_type(self):
        labels = bert.gate("yes", {**PREDICTED, "stance": "threat"})
        self.assertEqual(labels["threat_type"], "job_loss")


class SplitTests(unittest.TestCase):
    def rows(self):
        return [
            {
                "sample_id": f"W{i:05d}",
                "type": "entry",
                "entry_id": str(i),
                "period": "before" if i % 2 else "after",
                "stratum": "work",
                "work": "no" if i % 10 == 0 else "yes",
                "ai": "no",
            }
            for i in range(200)
        ]

    def test_only_work_texts_disjoint_and_held_out_left_out(self):
        parts = bert.split(self.rows(), held_out={("entry", "1")})
        ids = [row["entry_id"] for part in parts.values() for row in part]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertNotIn("1", ids)
        self.assertTrue(all(row["work"] == "yes" for part in parts.values() for row in part))
        self.assertEqual(len(ids), 179)
        self.assertGreater(len(parts["train"]), len(parts["test"]))

    def test_macro_f1(self):
        self.assertEqual(bert.macro_f1(["yes", "no"], ["yes", "no"], ("yes", "no")), 1.0)
        self.assertEqual(bert.macro_f1(["yes", "yes"], ["no", "no"], ("yes", "no")), 0.0)


if __name__ == "__main__":
    unittest.main()
