"""The candidate gate, exercised against failures it is supposed to catch.

The rule these tests cover was green for its entire life while doing nothing.
It forbade candidate evidence in the paper tree and carried one exception for
the ESR1 decomposability showcase -- and the showcase was not under any
directory the rule named, and carried none of the four strings it matched, so
the exception had never admitted anything. The gate passed because the file was
invisible to it, not because the exception let it through. The same probe found
``data/appendix_esr1/SHOWCASE_INPUT.json``: six molecules with their stability,
liability and metabolic verdicts, read by no completeness check at all.

That was found by removing the showcase from the registry and expecting the
gate to fail. It did not. So every clause below is checked by constructing the
input it must reject, on a tree built for the purpose, rather than by running
the verifier over the real repository and observing that it says "no
offenders" -- which it says identically whether the gate works or not.

The tests are cheap because ``candidate_showcase_checks`` was lifted out of
``main``. Before that the only way to run it was to run all sixty-odd checks
over two thousand files, which is why nobody ran it against anything but the
tree as it stood.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import verify_claims as vc  # noqa: E402

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"

AUDIT = {
    "stability_alerts": [],
    "liability_alerts": [],
    "metabolic_soft_spots": ["benzylic_position"],
    "n_unassigned_stereocentres": 0,
}

GLOBAL = {
    "total_record_cap": 40,
    "required_declarations": {
        "clinical_grade": False,
        "comparative_claim": False,
        "efficacy_or_affinity_claim": False,
        "repository_is_private": True,
    },
    "default_required_fields_on_every_record": [
        "candidate_id", "isomeric_smiles", "bond_graph_svg", "structural_audit",
    ],
    "what_structural_audit_must_carry": [
        "stability_alerts", "liability_alerts", "metabolic_soft_spots",
        "n_unassigned_stereocentres",
    ],
}


def _record(cid: str = "x-1") -> dict:
    return {
        "candidate_id": cid,
        "isomeric_smiles": ASPIRIN,
        "bond_graph_svg": "<svg/>",
        "structural_audit": dict(AUDIT),
    }


def _showcase(n: int = 2) -> dict:
    return {
        "schema": "test.showcase.v1",
        "clinical_grade": False,
        "comparative_claim": False,
        "efficacy_or_affinity_claim": False,
        "repository_is_private": True,
        "records": [_record(f"x-{i}") for i in range(n)],
    }


def _entry(**over) -> dict:
    e = {
        "path": "results/show/SHOWCASE.json",
        "schema": "test.showcase.v1",
        "record_cap": 12,
        "admitted_for": "a property, stated",
    }
    e.update(over)
    return e


class GateFixture(unittest.TestCase):
    """A two-file tree: one registry, one showcase. Each test bends one thing."""

    def build(self, registry: dict | None, showcase: dict | None,
              extra: dict[str, dict] | None = None) -> tuple[list[str], list[str]]:
        with TemporaryDirectory() as td:
            root = Path(td)
            files: list[Path] = []
            if registry is not None:
                p = root / "contracts/CANDIDATE_SHOWCASES.json"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(registry))
                files.append(p)
            if showcase is not None:
                p = root / "results/show/SHOWCASE.json"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(showcase))
                files.append(p)
            for rel, doc in (extra or {}).items():
                p = root / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(doc))
                files.append(p)
            return vc.candidate_showcase_checks(root, files)

    def registry(self, **over) -> dict:
        return {"schema": "geoaudit.candidate_showcases.v1",
                "global": dict(GLOBAL),
                "showcases": [_entry(**over)]}


class TestThePassingCase(GateFixture):

    def test_a_registered_complete_showcase_passes(self) -> None:
        offenders, problems = self.build(self.registry(), _showcase())
        self.assertEqual(offenders, [])
        self.assertEqual(problems, [])


class TestTheProhibitionCatchesMolecules(GateFixture):
    """The clause that was decorative. Each of these must produce an offender."""

    def test_de_registering_the_showcase_makes_it_an_offender(self) -> None:
        # This is the exact test that exposed the original bug: it passed.
        reg = self.registry()
        reg["showcases"] = []
        offenders, _ = self.build(reg, _showcase())
        self.assertEqual(offenders, ["results/show/SHOWCASE.json"])

    def test_a_molecule_at_an_unregistered_path_is_caught(self) -> None:
        offenders, _ = self.build(
            self.registry(), _showcase(),
            extra={"results/other/SOMETHING.json": {
                "schema": "unrelated.v1",
                "records": [{"candidate_id": "y-1", "isomeric_smiles": ASPIRIN}]}})
        self.assertIn("results/other/SOMETHING.json", offenders)

    def test_the_registry_must_match_the_schema_it_admits(self) -> None:
        doc = _showcase()
        doc["schema"] = "test.showcase.v2"
        offenders, problems = self.build(self.registry(), doc)
        self.assertEqual(offenders, ["results/show/SHOWCASE.json"],
                         "a path admitted for one schema must not admit another")
        self.assertTrue(any("schema is" in p for p in problems))

    def test_a_missing_registry_admits_nothing(self) -> None:
        offenders, problems = self.build(None, _showcase())
        self.assertEqual(offenders, ["results/show/SHOWCASE.json"])
        self.assertTrue(any("missing" in p for p in problems))

    def test_naming_the_field_without_a_value_is_not_caught(self) -> None:
        # The intended boundary. tools/emit_esr1_showcase.py names all three
        # identifier fields and must stay clean, or the gate fires on its own
        # generators and gets switched off.
        offenders, _ = self.build(
            self.registry(), _showcase(),
            extra={"results/other/GENERATOR_NOTES.json": {
                "fields_recomputed_here": ["isomeric_smiles", "inchi_key"],
                "what_this_is": "names the fields, holds no molecule"}})
        self.assertNotIn("results/other/GENERATOR_NOTES.json", offenders)

    def test_the_legacy_directory_rule_still_fires(self) -> None:
        # Widening a gate is when it stops catching what it used to.
        offenders, _ = self.build(
            self.registry(), _showcase(),
            extra={"evidence/kras/REPORT.json": {"what_this_is": "no smiles here"}})
        self.assertIn("evidence/kras/REPORT.json", offenders)


class TestCompletenessOfAnAdmittedShowcase(GateFixture):

    def _one(self, mutate) -> list[str]:
        doc = _showcase()
        mutate(doc)
        return self.build(self.registry(), doc)[1]

    def test_a_missing_required_field_fails(self) -> None:
        problems = self._one(lambda d: d["records"][0].pop("bond_graph_svg"))
        self.assertTrue(any("bond_graph_svg" in p for p in problems), problems)

    def test_each_audit_key_is_required(self) -> None:
        for key in GLOBAL["what_structural_audit_must_carry"]:
            with self.subTest(key=key):
                problems = self._one(
                    lambda d, k=key: d["records"][0]["structural_audit"].pop(k))
                self.assertTrue(any(k in p for p in problems for k in (key,)),
                                f"{key} may be dropped without failing")

    def test_an_audit_that_is_not_an_object_fails(self) -> None:
        problems = self._one(
            lambda d: d["records"][0].__setitem__("structural_audit", "clean"))
        self.assertTrue(any("structural_audit" in p for p in problems))

    def test_each_required_declaration_is_enforced(self) -> None:
        for key, want in GLOBAL["required_declarations"].items():
            with self.subTest(declaration=key):
                problems = self._one(
                    lambda d, k=key, w=want: d.__setitem__(k, not w))
                self.assertTrue(any(key in p for p in problems), key)

    def test_dropping_a_declaration_entirely_fails(self) -> None:
        # Absent must fail the same way as wrong. The ESR1 input file had all
        # four absent for its whole life and nothing said so.
        problems = self._one(lambda d: d.pop("repository_is_private"))
        self.assertTrue(any("repository_is_private" in p for p in problems))

    def test_a_showcase_with_no_records_fails(self) -> None:
        problems = self._one(lambda d: d.__setitem__("records", []))
        self.assertTrue(any("no records" in p for p in problems))

    def test_the_per_showcase_cap_is_enforced(self) -> None:
        offenders, problems = self.build(
            self.registry(record_cap=1), _showcase(n=3))
        self.assertEqual(offenders, [])
        self.assertTrue(any("exceeds its cap" in p for p in problems))

    def test_the_global_cap_binds_across_showcases(self) -> None:
        # Per-showcase caps do not bound the tree; ten showcases of twelve is a
        # dump assembled one contract diff at a time.
        reg = self.registry()
        reg["global"]["total_record_cap"] = 2
        _, problems = self.build(reg, _showcase(n=3))
        self.assertTrue(any("global cap" in p for p in problems), problems)

    def test_a_registered_showcase_absent_from_disk_is_not_an_error(self) -> None:
        # The registry may run ahead of the tree; that is not a violation.
        _, problems = self.build(self.registry(), None)
        self.assertEqual(problems, [])


class TestOverridingTheDefaultFieldList(GateFixture):
    """An input file carries identifiers and a verdict, not derived chemistry."""

    def _input_entry(self, **over) -> dict:
        e = _entry(
            required_fields_on_every_record=["candidate_id", "isomeric_smiles",
                                             "audit_verdict"],
            why_its_own_fields="it is the input, not the product",
            audit_field="audit_verdict")
        e.update(over)
        return {"schema": "geoaudit.candidate_showcases.v1",
                "global": dict(GLOBAL), "showcases": [e]}

    def _input_doc(self) -> dict:
        return {"schema": "test.showcase.v1", "clinical_grade": False,
                "comparative_claim": False, "efficacy_or_affinity_claim": False,
                "repository_is_private": True,
                "records": [{"candidate_id": "x-1", "isomeric_smiles": ASPIRIN,
                             "audit_verdict": dict(AUDIT)}]}

    def test_an_input_file_passes_without_a_bond_graph(self) -> None:
        offenders, problems = self.build(self._input_entry(), self._input_doc())
        self.assertEqual((offenders, problems), ([], []))

    def test_the_override_still_requires_the_audit_keys(self) -> None:
        # Which file a molecule sits in does not change the reasons to withdraw
        # it, so the four audit keys survive the override.
        doc = self._input_doc()
        doc["records"][0]["audit_verdict"].pop("liability_alerts")
        _, problems = self.build(self._input_entry(), doc)
        self.assertTrue(any("liability_alerts" in p for p in problems))

    def test_an_override_without_a_written_reason_fails(self) -> None:
        reg = self._input_entry()
        reg["showcases"][0].pop("why_its_own_fields")
        _, problems = self.build(reg, self._input_doc())
        self.assertTrue(any("why_its_own_fields" in p for p in problems))


class TestTheRealRegistry(unittest.TestCase):
    """The tree as it stands, checked against the contract it claims to follow."""

    def setUp(self) -> None:
        self.reg = json.loads(
            (ROOT / "contracts/CANDIDATE_SHOWCASES.json").read_text())

    def test_every_registered_showcase_states_what_it_demonstrates(self) -> None:
        for s in self.reg["showcases"]:
            with self.subTest(path=s["path"]):
                self.assertTrue(s.get("admitted_for"),
                                "a showcase with no stated purpose is a dump "
                                "with a cap")

    def test_the_esr1_input_is_registered(self) -> None:
        paths = {s["path"] for s in self.reg["showcases"]}
        self.assertIn("data/appendix_esr1/SHOWCASE_INPUT.json", paths,
                      "the six molecules that no gate read for the file's whole "
                      "life must stay registered")

    def test_the_visibility_check_records_how_it_was_established(self) -> None:
        v = self.reg["repository_visibility_confirmed_at"]
        self.assertEqual(v["result"], "private")
        self.assertTrue(v.get("how"), "a self-declared private repository is "
                                      "the unchecked claim the whole exception "
                                      "rests on")

    def test_the_tree_is_within_the_global_cap(self) -> None:
        total = 0
        for s in self.reg["showcases"]:
            p = ROOT / s["path"]
            if p.exists():
                total += len(json.loads(p.read_text()).get("records") or [])
        self.assertLessEqual(total, self.reg["global"]["total_record_cap"])


if __name__ == "__main__":
    unittest.main()
