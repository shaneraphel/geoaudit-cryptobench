"""No learned model anywhere on the detector's datapath.

The repository's central claim is that the shipped detector contains no trained
network and no fitted real-valued transform. That was true and unchecked, which
is a different thing from true and enforced.

The specific risk this guards is not carelessness. pLM-NN is rebuilt here and
reported as a baseline that beats us by 0.0243 on the official fold, and every
future session that reaches for that deficit will be tempted by the encoder
sitting in the cache. An embedding quietly entering a wire builder would not
break a test, would not change a schema, and would leave every number in the
paper looking the same while the sentence describing them stopped being true.

Two gates, matching the leakage firewall's shape:

1. Import gate -- no module under ``src/`` imports a learning framework or an
   embedding library.
2. Reference gate -- no line of code under ``src/`` names the cached encoder or
   its artifacts. A docstring may name ESM-2 to say it is excluded, which is
   what ``sequence_wires.py`` does and is worth keeping; the reference gate
   matches the concrete artifact names rather than the model family, so saying
   "ESM-2 embeddings are excluded by construction" stays legal and reading one
   does not.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# scipy is deliberately absent from this set and the omission is the point: it
# computes confidence intervals in the reporting path and never touches the
# datapath. A rule broad enough to catch "numerical package" would either admit
# every learning framework or ban the Wilson interval.
LEARNED = {
    "torch", "tensorflow", "jax", "flax", "keras", "esm", "transformers",
    "sklearn", "scikit_learn", "xgboost", "lightgbm", "catboost", "fairseq",
    "sentencepiece", "onnxruntime", "openvino", "tflite_runtime",
}

# Concrete artifacts rather than the model family, so prose about exclusion is
# still allowed and a read is not.
ARTIFACT_REFS = ("esm2_t36_3B", "PLMNN_WEIGHTS", "_plmnn", "ESM2_CACHE",
                 "PLMNN_SCORES", "PLMNN_SEQUENCES")


def _imports_of(py: Path) -> set[str]:
    mods: set[str] = set()
    for node in ast.walk(ast.parse(py.read_text())):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    return mods


def _code_lines(py: Path) -> list[str]:
    """Lines that are neither blank nor a comment.

    Docstrings are included on purpose. A module that reads an embedding and
    describes it in its own docstring has still read one, and the reference set
    is narrow enough that describing the exclusion does not trip it.
    """
    return [ln.strip() for ln in py.read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


class TestDetectorReadsNoLearnedModel(unittest.TestCase):
    def test_src_imports_no_learning_framework(self) -> None:
        offenders: dict[str, set[str]] = {}
        for py in sorted(SRC.rglob("*.py")):
            bad = {m for m in _imports_of(py) if m.split(".")[0] in LEARNED}
            if bad:
                offenders[str(py.relative_to(ROOT))] = bad
        self.assertEqual(
            offenders, {},
            f"the detector imports a learned model: {offenders}")

    def test_src_names_no_encoder_artifact(self) -> None:
        offenders: dict[str, list[str]] = {}
        for py in sorted(SRC.rglob("*.py")):
            bad = [ln[:80] for ln in _code_lines(py)
                   if any(ref in ln for ref in ARTIFACT_REFS)]
            if bad:
                offenders[str(py.relative_to(ROOT))] = bad
        self.assertEqual(
            offenders, {},
            f"the detector references an encoder artifact: {offenders}")

    def test_the_gate_would_catch_a_violation(self) -> None:
        """A gate that can only pass is not a gate.

        Both rules are exercised against text that should trip them, so a
        refactor that quietly stops matching anything fails here rather than
        reporting a clean detector forever.
        """
        tree = ast.parse("import torch\nfrom esm import pretrained\n")
        mods: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods.add(node.module)
        self.assertTrue({m for m in mods if m.split(".")[0] in LEARNED},
                        "the import rule matches nothing it should match")
        line = 'path = ESM2_CACHE / "esm2_t36_3B_UR50D.pt"'
        self.assertTrue(any(ref in line for ref in ARTIFACT_REFS),
                        "the reference rule matches nothing it should match")

    def test_naming_the_exclusion_in_prose_stays_legal(self) -> None:
        """sequence_wires.py says ESM-2 is excluded, and must keep being able to.

        The distinction the reference set draws is between naming a model
        family and naming a file. If this starts failing, the rule has been
        broadened into one that punishes documenting the constraint.
        """
        prose = ("psiblast, hmmbuild all absent). ESM-2 embeddings are "
                 "excluded by construction because they are a neural network.")
        self.assertFalse(any(ref in prose for ref in ARTIFACT_REFS))
        wires = SRC / "pocket_bench/methods/sequence_wires.py"
        self.assertIn("ESM-2", wires.read_text(),
                      "the wire module no longer states what it excludes")


if __name__ == "__main__":
    unittest.main()
