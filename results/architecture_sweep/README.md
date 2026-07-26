# Architecture search records

These files are the development record of the combinational fusion architecture
used by `algebraic_field`. **None of them is a headline result**, and the
distinction between them matters:

## The one file that may be used to justify a choice

| file | fold read | may inform the architecture? |
|---|---|---|
| `TRAIN_ONLY_SELECTION.json` | a cluster-disjoint half of the **training** fold | **yes** |

`TRAIN_ONLY_SELECTION.json` splits the 770 training units into two halves with
disjoint MMseqs2 clusters, compiles every candidate on one half and scores it on
the other. The winner named there — a parallel table bank with integer
multiplicity fusion followed by a multi-scale spatial counting gate — is the
architecture that ships. The official test fold is not read anywhere in that
computation.

## Files that read the test fold, and therefore may not

| file | what it measures |
|---|---|
| `FEATURE_CEILING_DIAGNOSIS.json` | per-invariant separation, and a closed-form Fisher direction over 6 / 12 / 20 / 35 rank features |
| `BANK_FUSION.json` | seven groupings of the 35 invariants into dense tables, fused by unweighted counting |
| `FULL_EXPANSION.json` | the complete pair (595 tables) and triple (6545 tables) counting expansion |
| `WIDE_BUS_BANK.json` | trading wires per table against levels per wire at fixed cell occupancy |
| `MULTISCALE_GATE.json` | one gate per radius, and all five radii summed, before and after fusion |
| `THRESHOLD_GATE.json` | uniform against integer-multiplicity fusion |
| `DUAL_TRACK_AB.json` | the earlier six-wire versus ten-wire quaternary field |

Each of these was computed on the 192-unit official test fold. Reporting the
best of them as the method's score would be selection on the evaluation set: the
number would carry the maximum of several noisy estimates and be optimistic by
an unmeasured amount. They are kept because they are the honest record of what
was tried and because the negative ones are informative — the complete pair and
triple expansion is *worse* than a six-table bank, and finer per-wire
quantization is monotonically worse at fixed occupancy — but the architecture
was not chosen from them.

The two searches agree. The train-only split ranks the nine candidates in the
same order as the test-fold sweep and selects the same winner, so the shipped
architecture is not an artifact of having looked at the test fold. That
agreement is the reason the test number can be read as an out-of-sample estimate.

## Reproducing

```bash
PYTHONPATH=src python3.12 tools/select_architecture_on_train.py   # the selection
PYTHONPATH=src python3.12 tools/diagnose_feature_ceiling.py       # the diagnosis
PYTHONPATH=src python3.12 tools/run_bank_experiment.py
PYTHONPATH=src python3.12 tools/run_full_expansion.py
PYTHONPATH=src python3.12 tools/run_wide_bus_bank.py
PYTHONPATH=src python3.12 tools/run_multiscale_gate.py
PYTHONPATH=src python3.12 tools/run_threshold_gate.py
```

All of them read the cached invariants
(`data/cryptobench_apo/_cascade_cache_{train,test}.npz`, built by
`tools/build_cascade_cache.py`), so a rerun costs seconds rather than the hours
that feature extraction takes.
