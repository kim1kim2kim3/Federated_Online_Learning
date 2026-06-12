import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class FedOSTCPhase4StatusAndSourcesTests(unittest.TestCase):
    def test_todo_marks_phase4_through_phase8_complete(self):
        text = (ROOT / "FEDOSTC_TODO.md").read_text(encoding="utf-8")
        phase4 = text[text.index("### Phase 4"):text.index("### Phase 5")]
        phase5 = text[text.index("### Phase 5"):text.index("### Phase 6")]
        phase6 = text[text.index("### Phase 6"):text.index("### Phase 7")]
        phase7 = text[text.index("### Phase 7"):text.index("### Phase 8")]
        phase8 = text[text.index("### Phase 8"):text.index("## 구현 리스크")]

        self.assertIn("Status: **완료**", phase4)
        self.assertIn("Status: **완료**", phase5)
        self.assertIn("Status: **완료**", phase6)
        self.assertIn("Status: **완료**", phase7)
        self.assertIn("Status: **완료**", phase8)
        for phase in (phase4, phase5, phase6, phase7, phase8):
            self.assertNotIn("- [ ]", phase)

    def test_fedostc_server_source_avoids_forbidden_paths(self):
        text = (ROOT / "fl-server" / "Fedostc.py").read_text(encoding="utf-8")
        for forbidden in (
            "client_oa",
            "eval_drift",
            "kl_threshold",
            "GATConv",
            "x_attr",
            "y_attr",
            "BaseFLServer.local_execute",
            "super().local_execute",
        ):
            self.assertNotIn(forbidden, text)
        for pattern in (
            r"\bselect_clients\s*\(",
            r"\.selected\b",
            r"\bselected_(?:clients|client_ids|ids)\b",
            r"\bw_pred\s*\[\s*tau\s*\]",
        ):
            self.assertIsNone(re.search(pattern, text))

    def test_run_registry_contains_fedostc(self):
        text = (ROOT / "run.py").read_text(encoding="utf-8")
        self.assertRegex(text, r'"fedostc"\s*:\s*FedOSTC')

    def test_phase5_decision_records_rho_source_ambiguity(self):
        text = (ROOT / "FEDOSTC_PHASE5_DECISION.md").read_text(encoding="utf-8")
        self.assertIn("Eq. (14) ambiguity", text)
        self.assertIn("post-local-update", text)
        self.assertIn("not a claim", text)


if __name__ == "__main__":
    unittest.main()
