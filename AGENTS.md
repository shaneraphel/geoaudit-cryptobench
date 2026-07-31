# Working in this repository

This file exists because the same few mistakes keep happening here, and each one
below was made in this repository, was caught, and is recorded in an artifact or a
commit you can go and read. None of it is general advice about being careful. Every
rule names the incident that produced it, because a rule without an incident behind
it gets skipped and a rule with one does not.

The repository's whole claim is that its numbers can be checked by someone who does
not trust us. That property is not maintained by intention. It is maintained by
gates that run in `make verify`, by 795 tests, and by the habits below.

## Before you believe a number you just produced

**A query that returns zero has not told you there is nothing there.** The RCSB
field `rcsb_entry_info.experimental_method` takes the value `"EM"`. Spelling it
`"Electron Microscopy"` returns `total_count: 0` and no error, and that zero was
briefly read as "there are no cryo-EM structures past the cutoff, this axis is
dead". There are 15,439. Before concluding that a set is empty, run the query with a
value you know is populated and confirm the harness works at all.

**A request that hangs is not always the network.** The same probe asked for 20,000
rows per page where the working inventory query asks for 10,000. RCSB does not
reject the larger page; it stops responding. Ninety minutes went into retry logic,
TLS diagnostics and a theory about the VPN before anyone compared the two requests
side by side. When something new hangs and something old does not, diff the two
requests before diffing the network.

**Two splits are not twelve.** On two cluster-disjoint halvings the anisotropic lift
rose monotonically with radius and 20 Å was best. On twelve it peaks at 14 Å and 20 Å
is behind it. The monotone reading was stated in prose before the full run finished
and had to be withdrawn. Smoke tests establish that code runs. They do not establish
which configuration wins, and a smoke test's ordering should not be repeated to
anyone as a finding.

**An estimate that is off by two orders of magnitude usually has a wrong
denominator, not a wrong constant.** Cryo-EM's viability as a second external set
was estimated at six units by multiplying entry counts by the X-ray yield of one
unit per three hundred entries. The real pool is 461 candidate accessions, because
an X-ray entry carries 1.9 protein chains and a cryo-EM entry carries 13.4 — they
are large assemblies containing many distinct proteins. When an extrapolation
crosses between two populations, check that the per-unit ratio transfers before
trusting the product. Counting is usually cheap; this count took sixty seconds once
the pagination bug was fixed.

## Before you believe a screen

**`clinical_grade` is `false` in every artifact and that is not decoration.** No
number here supports a claim about binding affinity, a molecule, a patient or a
therapeutic decision. If you are writing text that will be read outside this
repository, that constraint travels with the number.

**Check whether the ceiling is a ceiling.** Several tools screened ideas by fitting
a Fisher discriminant and reasoning that no counting field could exceed it, since
the solve gets arbitrary real coefficients over unquantised inputs. Measured under
one gate on the same splits and wires, the counting field beats the solve by +0.0053
and does so on 11 of 12 splits (`IS_FISHER_A_CEILING.json`). The argument had a hole:
a counting field is not linear in the wires, it is linear in the indicator functions
of quantised table cells, which represents interactions no linear solve over raw
wires can express. A bound that was argued and never measured is a correlate. Say so
when you use one.

**Measure against the thing that ships, not against its ancestor.**
`OPERATOR_BANK_CEILING.json` reported +0.0081 from 267 generated descriptors, on
12/12 splits, and it was the standing reason to pursue that direction. The baseline
was 35 local invariants; the deployed detector reads 645 wires. Against 645 the same
descriptors give +0.0005 at best and −0.0009 pooled
(`WIDE_BANK_CEILING.json`). The earlier lift was mostly the spatial expansion
arriving by a second route. Before running a comparison, name the arm that
corresponds to what is actually deployed, and if there isn't one, that is the bug.

## Before you write a number into prose

**Name which method.** Three detectors here are routinely confused because they
share a feature set. `geometric_foundation` has no fitted quantity at all and scores
0.6646. `algebraic_field` compiles integer counts from the training fold and scores
0.7667. `table_field` does the same at a different table topology and scores 0.7992
internally, 0.8411 externally. Against P2Rank's 0.7933 those are −0.1287, −0.0266 and
+0.0058 respectively. A sentence about "our training-free geometric detector" that
quotes 0.84 is describing two different programs at once. Paper §"Three classes of
detector" and README lines 12–20 exist because an earlier version of that paragraph
did exactly this.

**Report "not measured" as not measured.** The Set B probe printed
`0 clusters remain` when the truth was that the cached UniRef50 mapping covers Set A's
X-ray accessions and none of the 461 candidates, so the cluster count was absent
rather than zero. As printed it read as an empty axis, the opposite of what had been
found. If a number is unavailable, print the reason and what to run; do not print a
default and let it be mistaken for a measurement.

**State the direction a bias cuts.** P2Rank predicted no pocket on `8ylg_A`, which
makes its precision 0/0 and its MCC denominator zero, so that unit silently leaves
two of its own averages — removing its worst case and flattering it relative to us.
That is in the paper, not a footnote, because the reader cannot see it from the
numbers.

## Before you touch the frozen parts

**The external set is spent.** `results/external/EXTERNAL_SET.json` was frozen and
hashed in one commit, the preregistration pinning that hash landed in the next, and
the read tool verifies by `git merge-base` that the plan is an ancestor of HEAD. The
plan forbids changing any architecture, threshold, quantisation rule or partition
library in response to that read. Scoring a newly improved method on it does not
produce a second confirmatory result; it destroys the first one. A second set has to
be built and frozen before the method that will be read on it is finalised.

**Do not modify a tool that feeds a frozen artifact for an unrelated reason.**
`external_inventory.py` builds Set A. Its retry count is too low for a bad link, and
raising it cannot change a result — but the change was made in the probe by
monkey-patching rather than in that file, because minimal blast radius near a frozen
artifact is worth more than tidiness. Check what pins what before editing anything, because the
coverage is not uniform: `EXTERNAL_SET.json` carries a `code_sha256` over
`build_external_set.py` alone, while `TABLE_FIELD.json` carries one over eight files
— `table_field.py`, `table_bank.py`, `wide_descriptors.py`,
`expanded_descriptors.py`, `algebraic_descriptors.py`, `density_topology.py`,
`geometric_foundation.py` and `sequence_wires.py`. A comment fixed in `table_bank.py`
invalidates the compiled field.

**`git add -A` is not a review.** One commit swept 600 files and 292 MB of vendored
P2Rank binaries into the repository, including a 54 MB model over GitHub's file
limit, and the 292 MB is still in the history. Look at `git status` before staging,
and if a commit is about one experiment, its diff should be about one experiment.

**Never put an absolute path in an artifact.** A diagnostic string in
`EXTERNAL_SET.json` carried somebody's home directory, spelled out from the
filesystem root, because a writer named its destination in an error message. The builder now strips the checkout root; a reader
cannot use somebody's home directory and the gate
`no_local_absolute_paths_in_primary_docs` now fails on it. The one surviving instance
is exempted by name in `tools/verify_claims.py` because rewriting the artifact would
change its hash and break the confirmatory claim.

## What must be true before you commit

`make verify` green and `python3.12 -m pytest tests -q` green — currently 795 tests
and 553 subtests. New artifacts must be registered by `tools/classify_artifacts.py`,
which `make verify` checks; a new file under `results/` that is not in
`ARTIFACT_MANIFEST.json` fails the build. Any artifact produced on training folds
must say `"reads_test_fold": false` and mean it.

Commit messages here carry the reasoning, not a summary of the diff. If a result is
negative, the message says so in its first line; several of the most useful commits
in this history are ones where the idea failed and the failure was recorded precisely
enough to redirect the next attempt.

## What is deliberately not in this repository

No hardware-design pipelines, netlist tooling, accelerator sources, or the private
learning engines developed in the sibling repositories. The gate
`no_proprietary_engine_names_public` in `tools/verify_claims.py` holds the list of
names and fails any primary document that prints one, so consult the gate rather
than restating the list here — the first version of this section named all five and
failed its own check, which is a neat demonstration that a rule written carelessly
can leak the thing it protects. The judgement about what counts is yours. The
vendored P2Rank distribution is also excluded — `bin/install-p2rank.sh` reinstalls the pinned 2.5.1, and what makes
the baseline auditable is the archived CSVs, the recorded command, the version
banners and the per-file digests, all of which are tracked.

Business and IP material — CRO protocols, patent claim structure, licensing
checklists — does not belong here either, and not only for tidiness. Publishing the
coordinates of a cryptic pocket on a named high-value target can become prior art
against your own composition claims in absolute-novelty jurisdictions. Keep that
material in a separate private repository and settle the filing order before this one
names a target.
