import unittest

from amc.wykop import agent_annotation as agents

NO_WORK = {"ai": "yes", "work": "no", **dict.fromkeys(agents.DETAILS, "none")}
WORK_NO_AI = {
    "ai": "no",
    "work": "yes",
    "job_insecurity": "yes",
    "insecurity_cause": "economy",
    "sector": "it",
    "stance": "none",
    "attitude": "none",
    "threat_type": "none",
    "competence_claim": "none",
}
AI_WORK = {
    **WORK_NO_AI,
    "ai": "yes",
    "insecurity_cause": "ai",
    "stance": "threat",
    "attitude": "negative",
    "threat_type": "hiring",
    "competence_claim": "fewer",
}


class ValidateTests(unittest.TestCase):
    def test_valid_answers(self):
        for row in (NO_WORK, WORK_NO_AI, AI_WORK):
            self.assertEqual(agents.validate(row), row)

    def test_work_no_needs_none_details(self):
        with self.assertRaises(ValueError):
            agents.validate({**NO_WORK, "sector": "it"})

    def test_attitude_follows_ai(self):
        with self.assertRaises(ValueError):
            agents.validate({**WORK_NO_AI, "ai": "yes"})
        with self.assertRaises(ValueError):
            agents.validate({**AI_WORK, "ai": "no"})

    def test_cause_needs_insecurity(self):
        with self.assertRaises(ValueError):
            agents.validate({**WORK_NO_AI, "job_insecurity": "no"})


class DrawTests(unittest.TestCase):
    def rows(self, when, name, n):
        date = "2020-01-01" if when == "before" else "2024-01-01"
        return [
            {
                "type": "entry",
                "entry_id": f"{when}{name}{i}",
                "period": agents.period(date),
                "stratum": name,
                "text": "x",
                "text_hash": "h",
            }
            for i in range(n)
        ]

    def test_short_stratum_gives_rest_to_work(self):
        rows = [
            *self.rows("before", "ai_work", 3),
            *self.rows("before", "work", 50),
            *self.rows("before", "other", 50),
            *self.rows("after", "ai_work", 50),
            *self.rows("after", "work", 50),
            *self.rows("after", "other", 50),
        ]
        sample = agents.draw(rows, {"ai_work": 10, "work": 10, "other": 5})
        count = {}
        for row in sample:
            count[row["period"], row["stratum"]] = count.get((row["period"], row["stratum"]), 0) + 1
        self.assertEqual(count["before", "ai_work"], 3)
        self.assertEqual(count["before", "work"], 17)
        self.assertEqual(count["after", "ai_work"], 10)
        weights = {r["stratum"]: r["weight"] for r in sample if r["period"] == "before"}
        self.assertEqual(weights["ai_work"], "1.0000")
        self.assertEqual(len({row["sample_id"] for row in sample}), len(sample))

    def test_kappa(self):
        self.assertEqual(agents.kappa(["yes", "no"], ["yes", "no"]), 1.0)


if __name__ == "__main__":
    unittest.main()


class GoldTests(unittest.TestCase):
    def test_details_come_from_the_agent_who_said_work(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            agents.write_csv(
                folder / "batches/batch_001.csv",
                ("sample_id", "text"),
                [{"sample_id": "W1", "text": "a"}, {"sample_id": "W2", "text": "b"}],
            )
            key = [dict.fromkeys(agents.KEY_COLUMNS, "") | {"sample_id": s} for s in ("W1", "W2")]
            agents.write_csv(folder / "sample_key.csv", agents.KEY_COLUMNS, key)
            first = [{"sample_id": "W1", **AI_WORK}, {"sample_id": "W2", **AI_WORK}]
            second = [{"sample_id": "W1", **NO_WORK}, {"sample_id": "W2", **NO_WORK}]
            agents.write_csv(folder / "answers/a/batch_001.csv", agents.ANSWER_COLUMNS, first)
            agents.write_csv(folder / "answers/b/batch_001.csv", agents.ANSWER_COLUMNS, second)
            summary, disputes = agents.compare(folder, ("a", "b"))
            self.assertEqual({row["label"] for row in disputes}, {"work"})
            decisions = [
                {"sample_id": "W1", "label": "work", "final": "yes"},
                {"sample_id": "W2", "label": "work", "final": "no"},
            ]
            rows = {row["sample_id"]: row for row in agents.gold(folder, decisions, ("a", "b"))}
            self.assertEqual(rows["W1"]["stance"], "threat")
            self.assertEqual(rows["W1"]["insecurity_cause"], "ai")
            self.assertEqual(rows["W2"]["sector"], "none")
