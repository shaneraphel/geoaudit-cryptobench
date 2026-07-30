# Working memory for an agent that will forget this

`AGENTS.md` is the law of this repository and every rule in it names the incident
that produced it. This file is different and is not a second copy of it. It is
the running state of the investigation: what the numbers currently are, which
roads have been closed and by which artifact, what is queued, and which habits
have actually failed here rather than which habits are generally good.

It is written to be **edited every session**. A rule that is right forever
belongs in `AGENTS.md`. A number that will be wrong next week belongs here, next
to the artifact that would have to change for it to move. If you find a
statement here that the artifacts no longer support, the correct action is to
change this file in the same commit that changes the number, not to work around
it.

Nothing in this file is a claim about a molecule, a patient or a therapy.
`clinical_grade` is `false` in every artifact it refers to.

---

## 1. Where the numbers actually stand

Copy nothing from this section into prose without opening the artifact named
beside it. These are the four comparisons that get confused most often, and the
confusion is always the same one: three different detectors share a feature set
and get called by one name.

| Detector | What it fits | Official fold |
|---|---|---|
| `geometric_foundation` | nothing at all | 0.6646 |
| `algebraic_field` | integer counts on the training fold | 0.7667 |
| `table_field` | the same, at a different table topology | 0.7992 |

Against the baselines on the **official CryptoBench test fold** (192 units, read
12 times — the ledger counts them):

- **P2Rank 2.5.1**: 0.7933. Our margin is **+0.0058 and crosses zero**. Parity,
  not a win. Say parity.
- **pLM-NN** (ESM2-3B, CryptoBench's own baseline, rebuilt and run here):
  **0.8235**, ahead of us by **0.0243** [−0.0465, −0.0033], winning on 111 of
  192 chains. This is the real gap and it is the thing to close.
- **PocketMiner**: we are ahead by +0.0468, and P2Rank is ahead of it by +0.0410
  too, so that gap is about two label definitions and not about method quality.

On the **frozen external set** (57 units, read once, `EXTERNAL_READ.json`):

- P2Rank: **+0.0443 [+0.0162, +0.0724]**, ahead on 42 of 57, survives Bonferroni.
  This one resolved. It is the paper's confirmatory result.
- pLM-NN: the deficit replicates at **−0.0340 [−0.0701, −0.0006]**.

So "beat SOTA" has two different meanings and only one is open. Against P2Rank
it is done and published. Against pLM-NN nothing measured so far closes it.

## 2. Roads that are closed, and the artifact that closed each

Do not reopen one of these without a reason that names the artifact and says
what is different this time. Each cost days.

| Direction | Result | Artifact |
|---|---|---|
| 267 generated descriptor families | **+0.0005** against the deployed 645-wire bank | `WIDE_BANK_CEILING.json` |
| Deformation-across-scale family | **−0.0004** | same |
| Anisotropic columns, bus widened to 774 | **−0.0007**, positive on 5/12 | `ANISOTROPIC_COUNTING_FIELD.json` |
| Anisotropic columns, bank extended instead | **+0.0010**, positive on 9/12 | `UNION_BANK_COUNTING_FIELD.json` |
| Anisotropy to a linear solve | **+0.0038**, 12/12 | `IS_FISHER_A_CEILING.json` |

Two things follow from that table and both are load-bearing.

**Attachment matters as much as content.** Widening the bus from 645 to 774
wires keeps only **296 of 5,152** existing pairings, because `partition_tables`
redraws every pairing at the new width. That redraw alone was worth more than
the columns. Any new column block must be measured under the *union* attachment
— old pairings held, new tables over the new columns only — and not only under
widening. `union_bank_counting_field.py` and `composition_wires.py` both
implement it; reuse one rather than writing a third.

**A Fisher solve is not a ceiling on the counting field.** It was used as a
screen for months on the argument that a linear solve over unquantised inputs
must dominate a table of counts. Measured under one gate on the same wires the
counting field beats it by **+0.0053 on 11 of 12 splits**, because the field is
linear in the indicator functions of quantised cells and represents interactions
no linear solve over raw wires can express. Use the solve as a cheap correlate
that separates "the information is absent" from "the construction does not
collect it". Never quote it as a bound.

## 2b. Eight measurements that together say the architecture is at a local optimum

Added after a session that closed four axes and refuted two of its own
hypotheses; extended by the two that closed the per-region idea this section used
to propose, and then by the one that closed the table topology. Read this block
before suggesting anything that reweights, reselects, requantises, subdivides or
reshapes the present construction, because all five have now been measured. The
readout is exhausted: every axis it has is at or past its optimum, and the only
untried direction left is the wires themselves — what a table reads, rather than
how the tables are built, addressed, chosen, weighted, subdivided or shaped.

| Measurement | Result | Artifact |
|---|---|---|
| Quantisation cut points | at optimum; resolving the tail is monotonically worse, −0.0041 to −0.0069 | `QUANTISATION_LADDER.json` |
| Pairing choice | seed noise; a reseed costs −0.0026 and selection costs −0.0028 | `SELECTED_PAIRINGS.json` |
| Appended column families | solve sees them, field does not, on two families of different character | `COMPOSITION_WIRES.json` |
| Integer rounding of multiplicities | free, cos = 0.9992 | `GRAM_CONDITIONING.json` |
| Bank size | not compressible; smooth in `K'`, no knee, −0.0324 at `K'`=52 | `BANK_TRUNCATION.json` |
| Per-region multiplicities | best arm +0.0005 on 6/12 over deployed, under the noise floor; +0.0054 on 12/12 over a *random* router | `HIERARCHICAL_MULTIPLICITIES.json` |
| Per-region gate weight | −0.0010 and −0.0003 against one global weight; four numbers is already too many | `GATE_WEIGHT_ROUTING.json` |
| Table width | interior optimum at 2; width 3 loses on 0/12 either matching, width 1 by −0.0053 | `TABLE_WIDTH.json` |

**Two hypotheses were raised and killed in the same session. Do not raise them
again without new evidence.**

*Conditioning orders the banks.* Proposed because the field beats a linear solve
on the deployed wires by +0.0053 and loses on every bank with columns appended.
Measured: the ordering by condition number is `interaction-selected, deployed,
another seed, union` and by accuracy is `deployed, union, another seed,
interaction-selected`. They do not agree and neither does the direction cosine.
Withdrawn. What survives is a fact and not an ordering: selection really does cost
447 of 5,152 independent directions.

*The bank is effectively fifty-dimensional.* Proposed because 76.5 % of the
scatter trace sits in the top 1 % of its directions, about 52 of 5,152, and
`cos(m, Δμ)` is 0.025 on every bank including the deployed one. That predicted a
detector two orders of magnitude smaller. Measured: accuracy is smooth and roughly
logarithmic in `K'` with no plateau, and at `K'`=52 the loss is 0.0324 — larger
than the whole pLM-NN deficit. The residual 23.5 % of the trace is spread over
thousands of directions that each contribute little and together contribute a lot,
and a spectrum cannot tell that from redundancy. Withdrawn.

**The per-region idea this section used to propose has now been tested, twice.**
The argument was that every parameter varied up to that point was a *global*
choice applied identically to all 5,152 tables, and that `cos(m, Δμ) = 0.025`
looks like one direction fitted to heterogeneous regions.

**Say which comparison, because these two artifacts have three baselines and the
answer differs by baseline.** Against a *random* per-chain router, routing on
chain size is worth +0.0054 on 12 of 12 splits, so the router is reading real
structure and this is not noise. Against the *deployed* global solve, the best of
twelve routed arms is +0.0005 on 6 of 12 — below the 0.0026 reseed floor, which
is to say nothing — and the other eleven are negative, down to −0.0148. An earlier
version of this table said the per-region correction "is worth about +0.005",
which reads as a gain over the deployed detector and is not what was measured.
The regional heterogeneity is real; converting it into accuracy did not happen,
and it cost 20,608 fitted numbers to find that out.

The gate-weight arm says the same thing at a hundredth of the price. Four numbers,
the one stage where a chain-level quantity multiplies the field: the routers beat
a random router by +0.0010 and +0.0017 on 7 and 9 of 12, and lose to a *single
global weight* by −0.0010 and −0.0003. The constraint is granularity, not absence
of signal. A correction that pays has to be cheaper per region than anything
tried, and no such rule has been proposed.

**A caution about the deployed arm in both of those.** Every fitted arm also
loses to the deployed `w = 1.0`, and that is not evidence that a declared constant
beats a compiled one. The pair `r = 18, w = 1.0` was chosen over
{14, 18} × {0.5, 1.0} by pick-half ROC-AUC on halvings of this same training fold
— `FINAL_READOUT_SELECTION.json` and `PAIRWISE_READOUT_SELECTION.json` record it,
and both of those in fact selected `w = 0.5`. So the deployed weight has seen
pick-half information that every fitted arm is denied, and the bias cuts in its
favour. When an arm loses to deployed, check what selected deployed before
reporting the loss as a property of the arm.

**The table topology has since been tested too, and it is also closed.** See the
width entry in Section 3: two is an interior optimum, three loses on 0 of 12
splits under either matching, and the reason is the same count-versus-resolution
trade the quantisation ladder found. That makes eight parameters measured and
eight at or past their optimum, and it exhausts the readout. Anything further has
to change the wires — what a table reads — not how the tables are built,
addressed, chosen, weighted, subdivided or shaped.

## 3. What is open, and why each is thought to be open

**Quantisation cut points.** Every wire is cut at within-chain quartiles, so the
extreme level is the extreme 25 % of a chain, and cryptic residues are 5.76 % of
the fold. Measured on the training cache, the extreme 2 % of the strongest wires
is about twice as rich as the extreme quartile containing it — `void` reaches
0.2934 against 0.1291, `angle_deficit~d6` reaches 0.2147 against 0.1006. A level
holding both a 29 % band and a 9 % band states one frequency for both and no
integer weight recovers the difference. Moving the cuts costs nothing: same 4
levels, same 16 cells, same 5,152 tables. `quantisation_ladder.py`.

**Table pairings.** All 5,152 are drawn from one seed. Selection by measured
interaction was tried and **lost to random** on the first split, with
anti-selection losing by more, so the criterion orders correctly but the greedy
matching strategy does not help. Greedy captures only ~24 % of the unconstrained
ideal weight because interaction concentrates on few wires and a matching may
use each wire once per round. Cell occupancy was checked and is not the cause.
The live hypothesis is selection optimism, and `selected_pairings.py` now
re-measures the chosen banks on the half they were not chosen from.

**Table width — closed, and it is an interior optimum.** Width 3 costs −0.0036 at
a matched cell budget and −0.0066 at matched rounds, on **0 of 12** splits either
way; width 4 costs −0.0127; width 1, a per-wire lookup with no interaction at all,
costs −0.0053, which is what the pairing is worth (`TABLE_WIDTH.json`). Two is a
peak with both neighbours below it, not the edge of a plateau. The mechanism is
the quantisation ladder's, seen along a second axis: median residues behind an
addressed cell runs 29,673 / 7,385 / 1,763 / 401 for widths 1 to 4, a factor of
four per step. `table_bank.py`'s docstring guessed that a width-3 cell would still
hold thousands at 235k residues — it holds 1,763 on a fit half, and that is not
enough. Two practical notes: `partition_tables` takes `width` already, so no
digest-pinned file needs editing, and it ends with a filter dropping groups of
fewer than two wires, so width 1 must be built locally and 645 wires give 322
pairs per round rather than 322 and a singleton.

**Chemical composition of the neighbourhood.** Four of 645 wires carry residue
identity. pLM-NN reads a language model of the sequence. `composition_wires.py`
counts which chemical class sits in which shell, at which walk distance on the
contact graph, and in which pairs among the contacts — 76 integer columns, no
database, no alignment, nothing fetched. First reading of the Fisher correlate
was **+0.0010 on 3/3** splits.

**The last linear object — measured, and it stays.** `integer_fanout` rounds one
ridge solve; everything else in the inference path is integer. The obvious move is
to replace it with a counting rule, and that was tried five ways
(`COMBINATORIAL_MULTIPLICITIES.json`). Deleting the off-diagonal of the scatter —
the only thing a per-table rule cannot see — costs **−0.0953 on 0/12**, about
thirty-six times the reseed floor, where every other sweep here moves by
thousandths. Standardised difference −0.0909, quartile bands −0.0913, sign alone
−0.0959. Which per-table statistic you use does not matter.

**And the reason matters more than the refusal.** The solved direction sits at
cosine **0.030, 0.023, 0.019, 0.036** to those four rules — nearly orthogonal to
every statement a table can make about itself. With `cos(m, Δμ) = 0.025` from
`GRAM_CONDITIONING.json`, the reading is consistent: **the multiplicities are
almost purely a decorrelation, not a weighting.** 5,152 tables come from repeated
random partitions of 645 wires, so each wire sits in ~16 tables and they restate
each other heavily; the solve is mostly the correction for that restatement. This
is also why the ridge was never cosmetic — the same repetition drove the direction
into the null space when the pool grew without it.

So a combinatorial replacement cannot be a better per-table score; none can exist,
because the target is nearly orthogonal to all of them. It would have to be an
**integer decorrelation of the bank**, which is a different object from anything
tried and is the one open item on this axis. Do not propose per-table weighting
schemes here again.

## 4. Habits that have actually failed in this repository

These are not general advice. Each one happened, in this tree, recently.

**Reproduce the frozen arm before believing the new one.** A tool written this
session reimplemented the Fisher solve locally and did not standardise the
columns. Its 645-wire arm scored 0.03 below the frozen one and it reported the
new columns at **−0.0096 on 0 of 12** splits. With the canonical solve imported
instead, the same columns give **+0.0010 on 3/3**. The negative was entirely an
artefact of the reimplementation, and it would have closed a live axis. Every
tool that reimplements a piece of the pipeline must run the deployed
configuration through the local copy and require the frozen numbers back.
`quantisation_ladder.py` does this and reproduced 12 of 12 exactly.

**Do not edit a file a digest pins.** `TABLE_FIELD.json` carries a
`code_sha256` over eight files including `table_bank.py`. Adding a parameter to
`chain_digits` would invalidate the compiled field for a reason unrelated to the
field. Copy the function into the new tool and say in the docstring that you did
and why.

**One split is not twelve, and the smoke test's ordering is not a finding.**
Stated in `AGENTS.md` because a monotone reading from two splits had to be
withdrawn at twelve. It nearly happened again this session with the pairing
arms. The one licensed exception is when the effect is an order of magnitude
larger than any split variation the tree has produced — the truncation curve's
−0.067 at `K'`=13 is safe from one split — and even then the qualitative reading
is quoted and the numbers are not called final until twelve.

**Write the falsification condition into the tool before you run it.** Both
hypotheses killed this session were killed by a sentence their own tool's
docstring contained in advance: the conditioning tool printed both orderings and a
boolean for whether they agreed, and the truncation tool said that a smooth
fall-off rather than a plateau would mean the trace concentration was misleading.
Neither result could be talked around afterwards. A tool that can only confirm is
not an instrument.

**Author in `/tmp`, copy into the repository, commit, then run.** Two tools were
authored, run to completion, and then deleted along with every other untracked
file by a failed workspace switch, while their artifacts survived. An artifact
whose tool is gone is not reproducible, which is the one property this repository
claims. One had to be re-authored from its own artifact's recorded specification
and required to reproduce it exactly before being committed.

**Stage, then verify, then commit.** Two commits in one session landed with
`make verify` red. The gate walks `git ls-files`, so a new file passes every
content check until the commit that tracks it. Separately, do not register an
artifact in the manifest while a run is still rewriting it — the digest pinned
will be the wrong one.

**Check your own SMARTS, regex and column specs against a case you know.** Five
instrument defects this session, each caught by a gate or a test rather than by
reading: an unstandardised discriminant that inverted the sign of a live axis; a
five-membered succinimide labelled as a six-membered glutarimide, which rejected
every correct molecule in a release; hydrogen-bond donors that excluded amide
nitrogen and acceptors that omitted carbonyl oxygen, so a molecule full of ureas
reported zero of both; a pattern that did not compile and was displayed beside
genuine hits; and a home-directory regex inside the tool written to remove
home-directory paths.

**Check that the repository you are about to create already exists.**
`foliation-transfer-atlas` had a GitHub repository, an empty one, and a local
tree of 14,929 lines whose `.git` was an empty directory, so `git rev-parse`
resolved to the enclosing checkout, which ignores `.local/`. Nothing was tracked
by anything for two days. Run `gh repo list` and `git rev-parse --show-toplevel`
before building a skeleton.

**Diff the two requests before diffing the network.** `git clone` of
`foliation-transfer-atlas` failed with `SSL_ERROR_SYSCALL`, which reads as a VPN
problem. The same host over the same protocol worked for the sibling repository
in the same second. The actual cause was that the remote repository was empty
and had no refs, which the API says plainly as HTTP 409.

**A figure joins the provenance chain or it does not go in.** An image dropped
into `figures/` fails `make verify`, and the gate wants four things, not one: the
filename must appear in `make_official_figures.py` as `FIGDIR / "..."`, the image
must be listed in `FIGURE_PROVENANCE.json` with its sha256, the artifacts it was
drawn from must be in `SOURCES` so a change in them fails the digest check, and
the caption must be recorded there and appear verbatim in `README.md`. The gate
exists because `fig_baseline_comparison.png` sat in the tree for weeks plotting a
non-cluster-disjoint pilot with method names from no manuscript, showing every
detector at zero. Build the caption from the artifact rather than typing it, and
keep backticks and straight double quotes out of it — the same string becomes a
LaTeX macro, where both come out as the wrong glyph.

**Writing a contamination down does not protect you from it.** `SETB_POOL.json`
records that RCSB's `experimental_method == "EM"` also returns electron
crystallography, and states the practical risk in as many words: MicroED entries
occupy the top of any resolution-ordered selection. Within the same session,
EMD-46871 was chosen as "the high-resolution cryo-EM example" by taking the best
resolution available, and it is `electronCrystallography` at 1.09 Å — the entry
title is *Structure of proteinase K from energy-filtered MicroED data*. Two
signals in the file itself would have caught it before the download: a 0.2677 Å
voxel over a 55x59x61 Å cell is a crystallographic unit cell, not a
single-particle box, and the axis order was 3/2/1 rather than 1/2/3. The lesson
is not "remember the contamination". It is that a filter recorded in prose is not
a filter: ask EMDB for `structure_determination.method` and assert on the value,
because the map header and the entry metadata will both answer and neither is
expensive.

## 5. The frozen sets, and the one irreversible mistake

`results/external/EXTERNAL_SET.json` was frozen and hashed in one commit, the
preregistration pinning that hash landed in the next, and the read tool verifies
by `git merge-base` that the plan is an ancestor of HEAD.

**Set A is spent.** Scoring an improved method on it does not produce a second
confirmatory result; it destroys the first one. There is no undo. If a
counterattack lands, it must be confirmed on a set that was frozen *before* the
method was finalised.

Set B has been costed and not built: the cryo-EM pool is **461 candidate
accessions**, not the six an earlier extrapolation guessed, because an X-ray
entry carries 1.9 protein chains and a cryo-EM entry carries 13.4. The order is
not negotiable — build, freeze, hash, preregister, *then* finalise the method,
*then* read once.

## 6. Direction of the mathematics

Wanted: pure combinatorial logic, geometry, algebraic number theory, spectral
geometry, harmonic analysis, computational geometry, arithmetic geometry,
non-Euclidean computation, high-performance integer work. Integer spectra, walk
and non-backtracking walk counts, orbit counts under finite rotation subgroups,
Toeplitz structure carried to three dimensions, GF(4) point counts, p-adic and
ultrametric valuations.

Not wanted: numerical analysis, floating-point approximation, automatic
differentiation, fitting by linear algebra or linear transformation.

The distinction is about what ships, not about what may be measured. A ridge
solve is allowed as a screen and is used as one; it is not allowed in the
inference path, and the one place it still touches the path — the integer
fan-out — is listed above as open work for exactly that reason.

Be generous with operators. Being timid has not been the failure mode here;
believing an unmeasured argument has been.

## 7. Publication boundary

Hardware-design pipelines, netlist tooling, accelerator sources and the private
learning engines developed in the sibling repositories do not enter a public
repository. Two gates in `tools/verify_claims.py` hold the actual lists —
`no_proprietary_engine_names_public` for the engine names and
`no_out_of_scope_paper_material` for the subject areas — and both fail any
tracked primary document that prints a term from them. **Consult the gates; do
not restate their contents here.**

That instruction has now been ignored twice, once in `AGENTS.md` and once in
this file, and the second time is worth recording because it also exposes a
timing trap. The first version of this section spelled out the excluded
categories, which put one of the forbidden terms into a document inside the
paper tree. `make verify` was green when it was written and red as soon as it
was committed, because the gate walks `git ls-files` and an untracked file is
invisible to it. **A new document passes `verify` until the moment it is
tracked.** Stage it, then run `make verify`, then commit.

Business and IP material stays in a separate private repository, and the reason
is not tidiness. Publishing the coordinates of a cryptic pocket on a named
high-value target can become prior art against your own composition claims in
absolute-novelty jurisdictions. Settle the filing order before this repository
names a target.

## 8. Sibling repositories

| Repository | What it is | State |
|---|---|---|
| `geoaudit-cryptobench` | this one; the benchmark paper | 25 gates, 614 tests, in sync |
| `foliation-transfer-atlas` | zero-tuning transfer to 5 oncology targets | committed and pushed; no transfer run yet |

The transfer atlas holds its own `AGENTS.md`, twelve gates and a three-grade
exposure design. Its next measurement is already named: for menin's `4GQ3`
pocket, AHoJ-DB records `pocket_rms` between 0.51 and 1.16 across all ten apo
chains, every one under the 2.0 Å cryptic threshold, so that pocket carries no
cryptic label and whether menin has one at all is an open question about the
target rather than about the implementation.
