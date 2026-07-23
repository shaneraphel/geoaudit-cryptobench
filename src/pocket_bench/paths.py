from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MANIFEST_PATH = ROOT / "data/manifests/MANIFEST.json"
CLUSTER_PATH = ROOT / "data/manifests/STRUCTURE_CLUSTER_LEDGER.json"
PREDICTION_INPUT = ROOT / "data/manifests/PREDICTION_INPUT_MANIFEST.json"
PREDICTION_INPUT_MANIFEST = PREDICTION_INPUT
STRUCTURE_CLUSTER_LEDGER = CLUSTER_PATH
SEEDS_PATH = ROOT / "configs/seeds.json"
LOCKED_HPARAMS = ROOT / "configs/pilot_hparams.json"
VALIDATION_OUT = ROOT / "results/pilot"
QUARANTINE = ROOT / "legacy/quarantine"
DCA_THRESHOLD_A = 4.0
CLINICAL_GRADE = False
STATUS_OK = "OK"
STATUS_EMPTY = "EMPTY"
STATUS_CRASH = "CRASH"
STATUS_TOOL_UNAVAILABLE = "TOOL_UNAVAILABLE"
GENOTYPES_PRIMARY = frozenset({"WT", "Y537S", "D538G"})
