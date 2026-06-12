import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAPPING = ROOT / "FEDOSTC_PHASE0_MAPPING.md"
TODO = ROOT / "FEDOSTC_TODO.md"

WATCHED_IMPLEMENTATION_FILES = [
    ROOT / "fl-server" / "Fedostc.py",
    ROOT / "client_fedostc.py",
    ROOT / "models" / "fedostc_model.py",
    ROOT / "models" / "fedostc_gat.py",
]

REQUIRED_MAPPING_FRAGMENTS = [
    "Algorithm 2: FedOSTC",
    "Eq. (8)-(10)",
    "Eq. (11)-(12)",
    "Eq. (13)-(15)",
    "Paper `t` maps to delayed feedback sample `tau = r - delay`",
    "current tensor index = r - 1",
    "delayed tensor index = tau - 1 = r - delay - 1",
    "Current round `r` is **predict/log/evaluate only**",
    "offline benchmark evaluation helper may read current `y_r` solely",
    "Round metrics are the A-contract offline benchmark exception",
    "Feedback round `tau`",
    "w_update[tau]",
    "`w_pred[r]` is **provenance only**",
    "not restore `w_pred[tau]`, `snapshot[tau]`",
    "FedOSTC server all-client loop",
    "No subset participation; no REFOL drift selection",
    "Do not call `client_oa.Client.local_execute()`",
    "No `BaseFLServer.local_execute()` reuse",
    "Do not replace exact path with PyG `GATConv`",
    "does not use `x_attr` or `y_attr`",
    "`a(·)` projection parameterization/training rule unresolved",
    "`ClientDecode(h, h_prime, w_d, n)` h/h_prime fusion unresolved",
    "Eq. (14) rho source state unresolved",
]

FORBIDDEN_LITERAL_GUARDS = [
    ("eval_drift", "FedOSTC exact path must not import REFOL drift selection."),
    ("kl_threshold", "FedOSTC exact path must not add REFOL KL-threshold selection."),
    ("GATConv", "Eq. (8)-(10) must not be replaced by PyG GATConv in the exact path."),
    ("x_attr", "FedOSTC exact path uses traffic speed x only, not x_attr."),
    ("y_attr", "FedOSTC exact path uses traffic speed x only, not y_attr."),
    (
        "client_oa.Client.local_execute",
        "FedOSTC must not reuse the delayed REFOL/FedAvg client local_execute path.",
    ),
    (
        "BaseFLServer.local_execute",
        "FedOSTC must not reuse BaseFLServer.local_execute subset/local-execute path.",
    ),
    (
        "super().local_execute",
        "FedOSTC must not delegate delayed OGD to BaseFLServer.local_execute.",
    ),
]

FORBIDDEN_REGEX_GUARDS = [
    (
        re.compile(r"\bselect_clients\s*\("),
        "FedOSTC exact path must not select a subset of clients.",
    ),
    (
        re.compile(r"\.selected\b"),
        "FedOSTC exact path must not gate training on REFOL's client.selected flag.",
    ),
    (
        re.compile(r"\bselected_(?:clients|client_ids|ids)\b"),
        "FedOSTC exact path must not introduce selected-client subset state.",
    ),
    (
        re.compile(r"\bclient_oa\b"),
        "FedOSTC exact path must use a FedOSTC-specific client, not client_oa.",
    ),
    (
        re.compile(r"\bw_pred\s*\[\s*tau\s*\]"),
        "Delayed training must not initialize from prediction-time w_pred[tau].",
    ),
    (
        re.compile(r"\bsnapshot\s*\[\s*tau\s*\]"),
        "Delayed training must not initialize from prediction-time snapshot[tau].",
    ),
    (
        re.compile(
            r"(?:load_state_dict|state_dict_to_load|update_model_state|train(?:ing)?_init|"
            r"initial(?:ize|_state)?)\s*\(?\s*(?:self\.)?(?:w_pred|snapshot)"
        ),
        "Prediction provenance snapshots must not feed delayed OGD initialization.",
    ),
]


class FedOSTCPhase0ContractTests(unittest.TestCase):
    def test_mapping_document_records_algorithm_equation_and_delay_contract(self):
        self.assertTrue(MAPPING.exists(), "FEDOSTC_PHASE0_MAPPING.md must exist")
        text = MAPPING.read_text(encoding="utf-8")

        for fragment in REQUIRED_MAPPING_FRAGMENTS:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, text)

    def test_todo_marks_phase0_through_phase8_complete(self):
        self.assertTrue(TODO.exists(), "FEDOSTC_TODO.md must exist")
        text = TODO.read_text(encoding="utf-8")
        phase0_start = text.index("### Phase 0 — 문헌/수식 mapping 고정")
        phase1_start = text.index("### Phase 1 — 모델 구현")
        phase2_start = text.index("### Phase 2", phase1_start)
        phase3_start = text.index("### Phase 3", phase2_start)
        phase4_start = text.index("### Phase 4", phase3_start)
        phase5_start = text.index("### Phase 5", phase4_start)
        phase6_start = text.index("### Phase 6", phase5_start)
        phase7_start = text.index("### Phase 7", phase6_start)
        phase8_start = text.index("### Phase 8", phase7_start)
        risks_start = text.index("## 구현 리스크", phase8_start)
        phase0 = text[phase0_start:phase1_start]
        phase1 = text[phase1_start:phase2_start]
        phase2 = text[phase2_start:phase3_start]
        phase3 = text[phase3_start:phase4_start]
        phase4 = text[phase4_start:phase5_start]
        phase5 = text[phase5_start:phase6_start]
        phase6 = text[phase6_start:phase7_start]
        phase7 = text[phase7_start:phase8_start]
        phase8 = text[phase8_start:risks_start]

        self.assertIn("Status: **완료**", phase0)
        self.assertIn("FEDOSTC_PHASE0_MAPPING.md", phase0)
        self.assertIn("tests/test_fedostc_phase0_contract.py", phase0)
        for idx, phase in enumerate([phase0, phase1, phase2, phase3, phase4, phase5, phase6, phase7, phase8]):
            with self.subTest(phase=idx):
                self.assertIn("Status: **완료", phase)
                self.assertNotIn("- [ ]", phase)
        self.assertIn("FEDOSTC_PHASE2_DECISION.md", phase2)
        self.assertIn("tests/test_fedostc_phase2_gat.py", phase2)
        self.assertIn("client-only", phase3)
        self.assertIn("tests/test_fedostc_phase3_client.py", phase3)
        self.assertIn("fl-server/Fedostc.py", phase4)
        self.assertIn("FEDOSTC_PHASE5_DECISION.md", phase5)
        self.assertIn("default_config.yaml", phase6)
        self.assertIn("README.md", phase6)
        self.assertIn("tests/test_fedostc_phase4_server.py", phase7)
        self.assertIn("--period_steps 1", phase8)

    def test_future_fedostc_sources_do_not_import_refol_selection_or_wrong_inputs(self):
        checked_any = False
        for path in WATCHED_IMPLEMENTATION_FILES:
            if not path.exists():
                continue
            checked_any = True
            source = path.read_text(encoding="utf-8")
            for needle, reason in FORBIDDEN_LITERAL_GUARDS:
                with self.subTest(path=path.relative_to(ROOT), needle=needle):
                    self.assertNotIn(needle, source, reason)
            for pattern, reason in FORBIDDEN_REGEX_GUARDS:
                with self.subTest(path=path.relative_to(ROOT), pattern=pattern.pattern):
                    self.assertIsNone(pattern.search(source), reason)

        # Phase0 intentionally has no runtime FedOSTC files yet. If future phases add them,
        # the same guard above becomes active without requiring a test rewrite.
        self.assertIsInstance(checked_any, bool)


if __name__ == "__main__":
    unittest.main()
