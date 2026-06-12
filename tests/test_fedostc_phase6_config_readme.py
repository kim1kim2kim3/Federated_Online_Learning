import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "default_config.yaml"
README = ROOT / "README.md"
TODO = ROOT / "FEDOSTC_TODO.md"


class FedOSTCPhase6ConfigReadmeTests(unittest.TestCase):
    def test_default_config_exposes_only_allowed_fedostc_keys(self):
        config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        raw_text = CONFIG.read_text(encoding="utf-8")

        self.assertEqual(config["agg_model"], "")
        self.assertIn('"fedostc"', raw_text)
        self.assertEqual(config["fedostc_encoder_hidden_size"], 64)
        self.assertEqual(config["fedostc_decoder_hidden_size"], 128)
        self.assertEqual(config["period_steps"], 288)

        for forbidden_key in (
            "fedostc_gat_negative_slope",
            "use_x_attr",
            "use_y_attr",
        ):
            with self.subTest(forbidden_key=forbidden_key):
                self.assertNotIn(forbidden_key, config)
                self.assertNotIn(forbidden_key, raw_text)

    def test_readme_documents_fedostc_adaptation_boundary_and_examples(self):
        text = README.read_text(encoding="utf-8")
        lowered = text.lower()

        required_fragments = [
            "fedostc",
            "delayed-label adaptation",
            "delayed-label feedback",
            "delayed scheduling",
            "future `y` leakage",
            "non-paper covariates",
            "REFOL selection",
            "subset participation",
            "`a(·)` projection parameterization/training",
            "decoder `h`/`h_prime` bridge/fusion",
            "FEDOSTC_PHASE5_DECISION.md",
            ".venv/bin/python run.py --agg_model fedostc",
            ".venv/bin/python run.py --agg_model fedostc --delay 1 --rounds 3",
        ]
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment.lower(), lowered)

        self.assertIn("정확한 저자 구현과 동일하다고 주장하지 않습니다", text)

    def test_phase6_through_phase8_done(self):
        text = TODO.read_text(encoding="utf-8")
        phase6 = text[text.index("### Phase 6"):text.index("### Phase 7")]
        phase7 = text[text.index("### Phase 7"):text.index("### Phase 8")]
        phase8 = text[text.index("### Phase 8"):text.index("## 구현 리스크")]

        self.assertIn("Status: **완료**", phase6)
        self.assertIn("Status: **완료**", phase7)
        self.assertIn("Status: **완료**", phase8)
        self.assertNotIn("- [ ]", phase6)
        self.assertNotIn("- [ ]", phase7)
        self.assertNotIn("- [ ]", phase8)
        self.assertIn("default_config.yaml", phase6)
        self.assertIn("README.md", phase6)
        self.assertIn("FEDOSTC_PHASE5_DECISION.md", phase6)


if __name__ == "__main__":
    unittest.main()
