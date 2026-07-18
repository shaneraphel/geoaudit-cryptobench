# Structural Novelty Screen (NOT a Patent / FTO Analysis)

**Status:** computational structural-dissimilarity indicator · `clinical_grade = false`

> **Critical scope statement.** This document is a **structural-dissimilarity
> screen** — whole-molecule Morgan(2,2048) Tanimoto and Murcko-scaffold Tanimoto
> versus a small set of approved reference drugs. It is **NOT** a patent
> Freedom-to-Operate (FTO) analysis and does **NOT** prove originality,
> First-in-Class status, or non-infringement. Real FTO requires a registered
> patent attorney and a full patent-database (Markush) search. We report the
> numbers as-is, including where they fail our own novelty bar.

## Method
- Fingerprint: Morgan radius 2, 2048 bits (whole molecule) + Murcko scaffold.
- Reference set (approved drugs; illustrative, not exhaustive): gilteritinib,
  ponatinib, venetoclax, imatinib, dasatinib.
- Indicator bar (self-imposed, arbitrary): whole-molecule max Tanimoto **< 0.25**
  flags "structurally distinct from this reference set". This is a weak indicator,
  not evidence of patentability.

## Results (whole-molecule max Tanimoto vs reference set)

| Candidate | max Tanimoto | nearest reference | < 0.25 bar? |
|-----------|--------------|-------------------|-------------|
| FLT3 `cubane_carbox` | 0.188 | venetoclax | PASS |
| FLT3 `bcp_ext` | 0.224 | imatinib | PASS |
| ABL1 `cubane_diamide` | 0.223 | ponatinib | PASS |
| BCL2 `azaspiro_ext` | **0.426** | venetoclax | **FAIL** |

Murcko-scaffold Tanimoto (from the strict W3 novelty gate) additionally showed
the earlier W3 ABL1 `indazole_alkyne` at **0.644 vs ponatinib** — a clear analog,
which is why it was blacklisted.

## Honest reading
- FLT3 `cubane_carbox`, FLT3 `bcp_ext`, ABL1 `cubane_diamide`: **structurally
  distinct** from the reference set on this indicator (does not equal patentable).
- BCL2 `azaspiro_ext`: **fails** the < 0.25 bar (0.426) — it retains the
  venetoclax/navitoclax acylsulfonamide BH3-mimic warhead class. It should **not**
  be described as an original/First-in-Class chemotype.

## What this does NOT establish
- Not FTO, not non-infringement, not patentability.
- Reference set is tiny; a real screen needs ChEMBL/SureChEMBL/patent Markush DBs.
- Structural dissimilarity ≠ different mechanism or different IP coverage.
