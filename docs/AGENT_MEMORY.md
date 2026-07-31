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

## 2c. The wires, which were the only axis left, and are now closed too

Four column families now have both a counting-field lift and a linear-solve lift on
the same twelve splits under the same gate, and §2d has since measured the third and
last attachment. **The axis is closed; read §2d before proposing a fifth family.**

**Reopened, narrowly — read §2i and §2j.** The closure held for six families and
the account of *why* arrived later: every one of them was a function of data the
pipeline already reads. The seventh reads bytes it throws away — backbone atom
positions — and is worth +0.00196 against the deployed detector and +0.00267
against its own marginal-preserving permutation, both with intervals excluding
zero. So the sentence above is right about re-encodings and was wrong to
generalise from them to wires as such. It is still right that +0.002 does not
close a 0.0243 deficit.

| family | mean pairwise interaction | field | solve |
|---|---|---|---|
| deployed 645 wires | **+1.06e-05** | 0.79012 | 0.78483 (field ahead **+0.0053**) |
| composition 76 | +5.96e-06 | −0.0009 on 2/12 | +0.0011 on 10/12 |
| asymmetry 129 | **−9.69e-06** | +0.0010 on 9/12 | **+0.0038 on 12/12** |
| graph invariants 225 | +6.42e-06 | **−0.0061 on 0/12** | **−0.0027 on 1/12** |
| chemistry 42 | not screened | **+0.0002 on 7/12** | not measured |

**The fifth family, and the first one with a control arm** (`CHEMISTRY_WIRES_LIFT.json`).
`residue_chemistry.py` replaces `composition_wires.py`'s eight class labels with
fourteen chemical quantities per side chain — rotameric dihedrals, side-chain donors
and acceptors separately, formal charge at pH 7.4, pH-switchability, aromatic ring
atoms, carbon and polar-atom counts, β-branching, backbone flexibility and
constraint, metal-ligating and nucleophilic capability, Zamyatnin volume — over three
aggregations each.

The motivating quantity is `chi_rotatable` and the argument for it was good: a
cryptic pocket is closed and opens, side chains open it, the 645 deployed wires are
invariants of atom *positions* and cannot see conformational freedom, and the eight
classes cannot either, since the aliphatic class holds ALA/VAL/LEU/ILE/MET whose χ
runs 0, 1, 2, 2, 3. **The argument was good and the measurement is nothing:**
+0.00016 [−0.00065, +0.00098] on 7 of 12, inside the range committed before the run.

What makes it unambiguous is the control, which no earlier family had. Adding the
same 336 tables over the deployed wires with the family **absent entirely** gives
+0.00043 — as much or more. The small positive is bank size, not chemistry. **Every
future family must carry that arm**; without it this run reports a lift.

**The deployed wires are the only family the field reads better than a solve does.**
Two explanations for the other two are excluded, one by arithmetic and one by
measurement. Not an interaction the field misses: the solve is *linear*, so what it
extracts is a linear function of the raw columns. Not the field's estimation cost
either — varying an appended block's cell budget twenty-one-fold, 2,064 to 44,032,
moves the lift by a seventh of the reseed floor and in the wrong direction
(`APPENDED_BLOCK_GEOMETRY.json`).

**The screen is falsified in both of its forms. Do not screen a fifth family with
it.** The dimensionless version — mean of interaction over `V_pair`, the *share* —
was named before the run and fell first: it puts composition first, where the field
does worst, because composition's first-order term is a third of the others and the
denominator did the ordering. The **unnormalised** mean pairwise interaction ordered
the three correctly, but it was chosen *after* seeing that ordering, which made it a
hypothesis owed a prospective test. Graph invariants 225 was that test, its value
`+6.42e-06` and its predicted range were recorded before the run, and the run
refutes it two ways (`APPENDED_FAMILY_LIFT.json`):

- **Wrong on magnitude.** Predicted no material field lift, between −0.002 and
  +0.001. Measured **−0.0061 on 0 of 12**, which is more than twice the 0.0026
  reseed floor below the bottom of the range. A family can be actively harmful and
  no internal statistic of it said so.
- **Wrong on order, and exactly backwards.** By the unnormalised statistic the three
  appended families rank graph invariants, composition, asymmetry. By measured field
  lift they rank asymmetry (+0.0010), composition (−0.0009), graph invariants
  (−0.0061) — the reverse, on all three. Three points, so a reversal is one ordering
  in six by chance; suggestive of an inverted screen, not proof of one.

**And the category itself is gone.** What made "belongs to a solve" meaningful was
that the solve *gained* what the field did not, +0.0011 and +0.0038 on the two
earlier families. Here the solve loses **−0.0027 on 1/12** as well. The family is
collectible by neither readout, and the screen has no category for that. A linear
solve with 225 extra columns should be able to zero them; it does worse than without
them, which is a conditioning cost and not a missing signal.

**Where the interaction lives, which is the actionable part.** Splitting the deployed
bus's 207,690 pairs by whether they share a local quantity:

| pair kind | mean interaction | pairs |
|---|---|---|
| same quantity, different statistic | **−6.29e-05** | 4,515 |
| same statistic, different quantity | +1.08e-05 | 13,545 |
| different in both | +1.24e-05 | 189,630 |

Same-quantity is the only negative category and six times more negative than the
asymmetry family as a whole. **Reading one operator at several radii produces wires
whose joint says less than their marginals added**, which is the shape a solve
collects and a counting field cannot. That explains the asymmetry family completely:
129 columns from one operator at several radii *are* that category as a whole family.

So the design rule, and it is measured rather than argued: **a new family is worth
building only if its members are different quantities.** A parameter sweep of one
operator will not be collectible however good the operator is, and its lift will
appear in a solve. It also explains where the deployed bus's returns went — adding
statistics of existing quantities adds wires from the harmful category; adding
quantities does not.

## 2d. The cross term, and the arm nobody has built

The quantity §2c said was missing is now measured on the same fit half, under the
same decomposition, with the straddling block of the Gram formed directly and
required to reproduce `joint_counts` exactly — it does, to `0.0e+00`
(`COLLECTABILITY_SCREEN.json`, `interaction_with_the_deployed_bus`).

| family | internal | **cross with the 645-wire bus** | straddling pairs | field lift |
|---|---|---|---|---|
| deployed 645 wires | +1.06e-05 | — | — | (the bank) |
| asymmetry 129 | −9.69e-06 | **−1.24e-06** | 83,205 | **+0.0010** |
| composition 76 | +5.96e-06 | **+1.15e-05** | 49,020 | −0.0009 |
| graph invariants 225 | +6.42e-06 | **+1.23e-05** | 145,125 | **−0.0061** |
| graph invariants 15 | +5.05e-06 | **+1.93e-05** | 9,675 | not measured |

**Two facts, and they are the useful part of this whole line.**

*The cross term is larger than the internal one, for every family that has both a
positive value and a measured lift.* Composition's synergy with the bus is nearly
twice its synergy with itself, and it is **above the deployed bank's own mean
pairwise interaction of +1.06e-05** — the bank whose tables support a field that
beats a linear solve by +0.0053. Graph invariants sit just above it too.

*The union attachment forms none of those pairs.* `appended_family_lift.py` builds
`union = old + [[c + n_old for c in t] for t in new]`: 5,152 tables over old wires,
1,792 over new columns alone, and **zero straddling**. Widening does form them and
is not an alternative, because `partition_tables` redraws every pairing at the new
width and only 281 of 5,152 old tables survive. So the strongest collectible
structure these families have is the structure both existing attachments discard —
one by construction, the other by destroying the bank to reach it.

**The reading, stated so it can be wrong.** More synergy with the bus goes with a
worse field lift, exactly reversed, on the three families with both numbers. That is
what you would see if the synergy is real but unreachable: the attachment cannot pair
across, so a family whose value lives in the crossing contributes only its weak
self-pairs while diluting the fan-out's decorrelation. It is also what you would see
if high-cross families are simply bad families for an unrelated reason. Three points
cannot separate those, and **this statistic was computed after the lifts were known**,
which is the same mistake the internal statistic made. It is a correlate until an
experiment moves the mechanism rather than another family being screened by it.

**The experiment that moves it — run, and it closes the axis.**
A third attachment, never built here before: keep all 5,152 old tables unchanged and
add 608 tables that each pair one deployed wire with one new column, matched in count
to what the union arm adds, on composition 76. Twelve cluster-disjoint splits, a
`more_old` control adding the same 608 tables over the deployed wires with the family
absent, and the preregistered criterion above (`STRADDLING_ATTACHMENT.json`):

| comparison | mean | 95 % | splits | sign-test p |
|---|---|---|---|---|
| union − deployed | **−0.00086** | [−0.00150, −0.00022] | 2/12 | — |
| straddle − union | +0.00087 | [−0.00006, +0.00181] | 9/12 | 0.073 |
| straddle − `more_old` | +0.00066 | [−0.00047, +0.00179] | 9/12 | 0.073 |
| **straddle − deployed** | **+0.00001** | [−0.00093, +0.00095] | 5/12 | 0.806 |

**The mechanism is real and the size of it is nothing.** Straddling is above the union
arm on 9 of 12 and above the control on 9 of 12, so the direction the cross term
predicted is not noise — but both intervals cross zero, both are a third of the 0.0026
reseed floor, and the preregistered criterion asked for more than the floor. The
decisive row is the last one. The union attachment genuinely costs −0.00086, the one
comparison here whose interval excludes zero; straddling recovers **exactly** that
cost and lands at +0.00001 against the deployed detector. The crossing was worth
having only because the attachment was throwing it away. There is no gain.

**So the wire axis is closed, by the same standard as the other eight.** Five families
and now all three attachments have been measured. Nothing moves the deployed detector.
Two statistics ordered the families — internal interaction, then cross interaction —
and neither moved a number when acted on. **Stop screening families.** The fifth was
proposed on a chemical argument rather than on either statistic, which was the right
kind of reason and produced the same null; so proposing a sixth needs more than a good
argument, and proposing a fourth attachment needs a reason that is not "the tables sit
somewhere else".

**And the one procedural thing that did change: run the control.** `more_old` adds
the same number of tables over the deployed wires with the new family absent. On the
chemistry family it scored higher than the family did. Four earlier families were
measured without it, so their small positives and negatives cannot be separated from
bank size after the fact — asymmetry's +0.0010 in particular has no control beside it
and should not be quoted as though it does.

Keep the size in view: the largest effect anywhere on this axis is +0.0010, and the
pLM-NN deficit is **0.0243**, twenty-four times it. That gap is not an attachment
problem and it was never plausible that it was.

## 2e. The largest effect in this repository, and nobody had looked

Every sweep in §2b and every family in §2c was scored by a **mean over units**.
`FAILURE_TAIL.json` scores the deployed field per unit on the pick half of all
twelve splits and reproduces the frozen per-split numbers exactly on the way
through, so this is the same detector read a different way. Read that way it is
two different detectors.

| cryptic residues in the unit | units | mean AUC | median | sd | null se | sd/se | share below 0.5 |
|---|---|---|---|---|---|---|---|
| 0–9 | 188 | **0.5991** | 0.6238 | 0.2914 | 0.1110 | 2.63 | **34.0 %** |
| 10–15 | 180 | 0.8144 | 0.8731 | 0.1821 | 0.0852 | 2.14 | 9.4 % |
| 16–22 | 200 | 0.8544 | 0.8983 | 0.1332 | 0.0687 | 1.94 | 2.5 % |
| 23–76 | 201 | 0.8766 | 0.9024 | 0.0989 | 0.0566 | 1.75 | 0.5 % |

Overall the mean per-unit AUC is 0.7884 and the **median is 0.8712**. The worst
fifth of units sits at 0.4135 against 0.8823 for the rest and carries 46 % of the
total deviation; 87 units score below 0.5, which is worse than a coin toss on
their own residues.

**Both explanations are true and only one of them matters.** Within-unit AUC over
`n1` positives has a null sampling standard error of about
`sqrt((n1+n0+1)/(12·n1·n0))`, so a unit with eight positives is noisy whatever the
detector does, and the observed spread is 2.63 times that null at small `n1`
against 1.75 at large. That is real. But sampling noise is **symmetric**: it
inflates a standard deviation, it does not move a mean, and the mean falls from
0.8766 to 0.5991 across the strata. A quarter of the training fold is at 0.60 and
the rest is at 0.81–0.88.

**What this reframes.** The official fold's mean paired difference against P2Rank
is +0.0058 and crosses zero while its 20 % trimmed mean is +0.0281 at p=0.002, and
the two withdrawn claims — F1 and MCC — are both threshold metrics, which are more
sensitive at small `n1` than AUC is. A deficit concentrated in small-pocket units
would produce all three of those observations. Whether it does is a question about
the official fold and **must not be answered by looking at it**: the stratification
above is on training units, and repeating it on the test fold to see whether the
story holds is method development against held-out data.

**The mechanism to test first, and it is on the one axis §3 still lists as open.**
Every wire is cut at within-chain quartiles, so the extreme level is the extreme
25 % of a chain. In a 230-residue chain with 8 cryptic residues that level holds
57 residues for 8 positives — a seven-fold dilution — where a unit at the top
stratum has 40 positives in the same 57. §3 already records that the extreme 2 % of
the strongest wires is about twice as rich as the quartile containing it, and
`QUANTISATION_LADDER.json` found the cuts at optimum with finer cuts monotonically
worse by −0.0041 to −0.0069 — but that sweep was scored by the same mean over
units, so a cut that helps the bottom stratum and hurts the top would have
cancelled inside it.

**Re-run stratified, and the prediction is falsified** (`QUANTISATION_BY_STRATUM.json`,
deployed arm reproduced to 4.0e-07):

| arm | pooled | 0–9 | 10–15 | 16–22 | 23–76 |
|---|---|---|---|---|---|
| uniform quartiles (deployed) | 0.7884 | 0.5991 | 0.8144 | 0.8544 | 0.8766 |
| tails at 5 % | −0.0056 | **−0.0032** | −0.0065 | −0.0077 | −0.0047 |
| tails at 2 % | −0.0067 | **−0.0049** | −0.0072 | −0.0075 | −0.0070 |

Finer tails hurt **every** stratum. The small-pocket stratum is hurt least, which
is the predicted direction and is still a loss, so the dilution argument is sound
about the arithmetic and wrong about the cause. **The small-pocket deficit is not
a quantisation-resolution problem**, and the ladder is closed a second time by a
stronger measurement than the one that closed it first.

That elimination leaves the 0.5991-against-0.8766 gap standing and unexplained.

**Read §2h before acting on this section.** The gap is real and this section
measures it correctly, but the inference everyone drew from it — that a quarter of
the fold is sitting there as recoverable headroom — is refuted: an unrelated rival
scores 0.5985 on the same units. The size of the effect is not the size of the
opportunity, and two sweeps (§2f, §2g) were spent before anyone checked.

## 2f. The gate radius has no provenance, and 14 beats 18 everywhere

`GATE_BY_STRATUM.json` sweeps seven radii and two weights, per unit and per
stratum, sharing one compile per split so thirteen arms cost what one used to.
Deployed arm reproduced to 4.0e-07.

**The deployed gate was never selected for the architecture that ships.**
`FINAL_READOUT_SELECTION.json` chose `r=18, w=0.5` under `top_k_wires=90`,
`table_width=6`, `n_tables=15`, `ridge=1e-06` and a `lin_digits` linear readout.
What ships is 645 wires, width 2, 5,152 tables, a counting field — and `w=1.0`,
which that selection did not choose either. The radius is inherited from a
configuration sharing almost nothing with the current one, and nobody re-swept it.

**Paired against the deployed gate on the same units:**

| arm | pooled | 0–9 | 10–15 | 16–22 | 23–76 |
|---|---|---|---|---|---|
| `r=14 w=1` | **+0.00250** [+0.0006, +0.0044] | +0.0023 | +0.0024 | +0.0027 | +0.0025 |
| | 436/769 better, **p=1.15e-04** | crosses 0 | crosses 0 | excludes 0 | excludes 0 |

A **uniform** gain of about +0.0025, the same size in every stratum. So the
reading that "the optimum radius rises with pocket size" is *wrong*: `r=14` is
best for three of four strata and by the same margin in all four. It is not a
small-pocket fix, it is a better global radius, and it needs no label knowledge —
which is what makes it the only thing measured in this whole line that could ship.

**And the small-pocket gate idea is refused by the pairing.** Removing the gate
entirely looks like +0.0117 on the 0–9 stratum by stratum means, which is 4.5× the
reseed floor and reads as a lever. Paired, it is **+0.0117 [−0.0056, +0.0289] with
83 of 188 units better** — fewer than half. The stratum mean moved because a few
units moved a great deal, which is the small-`n1` variance of §2e biting the
analyst instead of the detector. Every other stratum is hurt and hurt decisively
(−0.0127, −0.0182, −0.0254, all excluding zero). **Compare arms paired or do not
compare them.**

**Two limits, and neither is optional when this is written up.**

*Selection optimism — measured, and it is present but does not eat the effect.*
`r=14, w=1` is the best of thirteen arms chosen on the same pick halves it is
scored on, so the remedy §3 records for pairings was applied: choose on six
splits, score on the other six, both ways round.

| chosen on | arm chosen | margin there | margin on the other half |
|---|---|---|---|
| first 6 | `r=14 w=1` | +0.00320 | +0.00175, CI [−0.0002, +0.0037] |
| second 6 | `r=14 w=1` | +0.00175 | +0.00320, CI [+0.0011, +0.0053] |

**Both halves independently choose the same arm**, so the choice is not an
artifact of where it was selected. The margin differs between halves, +0.0032
against +0.0018, and one of the two intervals crosses zero — which is what six
splits buy. The tool prints `margin survives: False` because it requires both
directions to exclude zero, and that boolean is too crude to quote on its own:
what the numbers say is that the *choice* replicates and the *size* is somewhere
between +0.0018 and +0.0032, pooled +0.0025.

*The magnitude sits on the reseed floor.* +0.0025 against a 0.0026 floor. The
paired p is 1.15e-04 and the floor is about pairing-seed noise rather than about
this comparison, so the two are not the same quantity — but a gain the size of the
noise on a different axis is a coincidence to state, not to argue past.

*And changing it spends nothing but costs something.* The gate is part of the
method. Moving it makes `EXTERNAL_READ.json`'s +0.0443 a result about a detector
that no longer exists; confirmation would have to come from Set B or Set C, which
are frozen and unread for exactly this situation.

One caution that will matter as soon as a fix is proposed. The stratifying
variable is the number of cryptic residues, which is the **label**. Nothing that
conditions on it can ship. A fix has to condition on something observable — chain
size separates only weakly here, P(tail>rest)=0.393, so it is not a substitute —
and finding that observable is the actual problem.

## 2g. The tail is a noise problem, not a dilution problem

`GATE_FORM.json`, 12 splits, 769 units. The gate's *form* had never been varied:
§2b's eight parameters are all about the table bank, and §2f swept only the gate's
radius and weight. The form is "add back the neighbourhood **mean**, rescaled to
the raw field's spread", and the argument for changing it was mechanical. An
18 Å ball around a residue holds a couple of hundred residues. If eight of them
are cryptic the mean barely moves; if forty are, it moves a lot. That is exactly
the shape of §2e's failure, so the aggregation looked like the cause and an order
statistic looked like the cure. Prediction written into the tool before the run:
`max` and `top-k` beat `mean` on the 0–9 stratum, the gap narrows on 23–76, and
`median` is worst everywhere.

**Every clause of that is wrong, and the way it is wrong is the useful part.**

| arm | pooled | 0–9 | 10–15 | 16–22 | 23–76 |
|---|---|---|---|---|---|
| `mean` r=14 | **+0.00250** | +0.0023 | +0.0024 | +0.0027 | +0.0025 |
| `median` r=18 | −0.00479 | **+0.0071** | −0.0051 | −0.0084 | −0.0120 |
| `top5` r=18 | −0.01216 | −0.0420 | −0.0105 | **+0.0012** | **+0.0010** |
| `max` r=18 | −0.01549 | **−0.0376** | −0.0151 | −0.0040 | −0.0066 |
| `count` r=18 | −0.02546 | −0.0553 | −0.0225 | −0.0118 | −0.0139 |

Order statistics do not help the small-site stratum. **Their entire cost is
incurred there**: on the two largest strata `top5` is very slightly *ahead* of the
deployed mean (+0.0012, +0.0010) and `max` is behind by under 0.007, while on 0–9
both are behind by about 0.04. The predicted magnitude ordering is exactly
inverted, which is more informative than a null would have
been: it says the neighbourhood's peak is *less* trustworthy when there are few
positives, not more. With eight positives the largest score in the ball is
usually a high-scoring negative; with forty it is usually a real one. So the
0–9 stratum is **noise-limited, not dilution-limited**, and the gate earns its
keep by averaging noise away rather than by finding a peak.

`median` is the one arm that agrees with that reading rather than the old one: the
most outlier-robust form, +0.0071 on 0–9 with a CI of [+0.0011, +0.0131], and a
loss on every other stratum that grows monotonically with pocket size. **Do not
quote that as a win.** The sign test is 96/188, p=0.41 — the stratum mean moved
because a few units moved a lot, which is precisely the small-`n1` variance §2f
already caught doing this to an analyst. It is a direction, not a result.

Two things follow, and both are recorded as untried rather than promising:

* A **trimmed neighbourhood mean** is the form this reading actually predicts —
  averaging (which the mean has and the max lacks) plus outlier robustness (which
  the median has and the mean lacks). Neither swept arm has both. If the reading
  is right it beats both; if it lands between them the reading is decoration.
* Nothing here can be selected per stratum, for §2f's reason: the stratifying
  variable is the label. A blend of two gate forms is deployable; a switch
  between them is not.

The reproduction check matters more than usual here, because every arm is a
reimplementation of a pinned function: `mean` at the deployed radius reproduces
the frozen per-split numbers to **4.01e-07**, so the differences above are the
aggregation and nothing else. `table_field.py` was not touched — it is one of the
eight files under `TABLE_FIELD.json`'s `code_sha256`, and a local copy of the gate
costs nothing while an edit would invalidate the compiled field.

## 2h. The tail is not headroom. A rival collapses on it too, by the same amount

`BASELINE_BY_STRATUM.json`, 752 units where both methods cover the same residues
and agree on the positive count. §2e made the small-site stratum look like the
whole game — a quarter of the fold at 0.5991 against 0.8766, carrying 46% of the
total deviation, and arithmetically enough to cover the 0.0243 deficit against
pLM-NN twice over. §2f and §2g then spent two sweeps trying to fix it. The
question nobody had asked first is whether anyone else can do better there.

| stratum | ours | PocketMiner | ours − PocketMiner |
|---|---|---|---|
| 0–9 | 0.5958 | **0.5985** | **−0.0027** [−0.0355, +0.0300] |
| 10–15 | 0.8149 | 0.7637 | +0.0512 [+0.0298, +0.0726] |
| 16–22 | 0.8550 | 0.7976 | +0.0573 [+0.0415, +0.0732] |
| 23–76 | 0.8754 | 0.7993 | +0.0761 [+0.0620, +0.0903] |
| pooled | 0.7873 | 0.7412 | +0.0461 [+0.0349, +0.0572] |

**On the stratum that looked like the target, the two methods are indistinguishable
and both are near chance.** PocketMiner is a graph network over structure and we
are a counting field over quantised invariants; they share no architecture, no
featurisation and no fitting procedure, and they land 0.0027 apart. Its own fall
from its largest stratum to its smallest is −0.2008 against our −0.2796, so the
collapse is not a quirk of one design.

**Both biases in this comparison flatter PocketMiner**, which is what makes the
reading survive a hostile test rather than a friendly one. Its training set was
never clustered against CryptoBench's folds and six training entries match its own
by exact PDB id, which is a floor and not a homology check; and it drops residues
it cannot featurise, so on some units it is scored on an easier universe — the 17
units where that happens are excluded from the table above and included in the
artifact's second block. It is advantaged, and it still cannot beat 0.60 there.

**So the tail is not headroom, and the operators should not be built for it.**
A unit with seven cryptic residues in three hundred is an ambiguous labelling
problem before it is a detection problem, and 0.60 is roughly what a good method
gets. §2e's sentence naming it "the largest effect in this repository" is correct
about the *size* of the effect and was about to be wrong about what to do with it.

Two things this does *not* say, both of which will be tempting:

* It does not say the stratum is at its information limit — two structural
  methods can share a blind spot. It says no evidence of headroom exists, which
  is the standard §2c applied to five wire families.
* It does not locate the pLM-NN deficit. pLM-NN reads evolutionary information
  that neither method here has, and whether *its* profile also collapses on 0–9
  is unmeasured. That is now the question worth the compute: if pLM-NN also sits
  near 0.60 there, its whole advantage lives in the large strata, where we are
  already 0.076 ahead of PocketMiner, and that is a very specific target. This
  costs no test-fold read — the baseline can be run on the training fold, and
  nothing pins its tools by `code_sha256`.

Method note worth keeping: this cost **zero** compute on the rival, because its
per-residue training-fold predictions were already on disk under
`data/baselines/pocketminer_train/`. The tempting version of this measurement was
to stratify the *test-fold* pLM-NN comparison, which would have spent a read from
the ledger on an exploratory question. Check what is already on disk before
spending a frozen resource.

## 2i. Why six families were null, stated as a rule you can apply before running

Six wire families have now been measured and every one is null: graph invariants
(§2c), the operator bank, composition, the expanded and wide banks, and chemistry
42 (`CHEMISTRY_WIRES_LIFT.json`: union +0.000165 crossing zero, straddle −0.000662,
and the `more_old` control **ahead of both at +0.000430**). Six nulls with no
account of why is a habit; with an account it is a screen. The account:

> **Every one of those families is a function of data the deployed pipeline
> already reads. Before proposing a family, ask whether it reads bytes the
> pipeline currently throws away. If it does not, it is a re-encoding, and six
> measurements say re-encodings are worth nothing.**

Chemistry 42 is the cleanest case and it is worth having the arithmetic, because
the argument is checkable rather than rhetorical:

* The bank already carries seven constants that are functions of residue type —
  `kd`, `volume`, `aromatic`, `charge`, `hbd`, `hba`, `chi`. **Unquantised, those
  seven are injective on the twenty types**; `kd` and `volume` alone already are.
  So residue identity is fully determined by wires that are already deployed, and
  all fourteen chemistry columns are functions of residue type. Nothing about type
  can be new.
* **Quantised**, they are not injective. At four bands the seven resolve **17 of
  20 types**, and the three collisions are `ALA/GLY`, `ARG/LYS`, `ILE/LEU`.
  Summing over width-2 tables loses nothing further — separability under a sum
  over all pairs is separability wire by wire — so 17 is what the deployed field
  sees.
* So there *was* a representational gap, it was nameable, and chemistry 42 fills
  it: `sc_carbon`, `backbone_flexible` and `sc_volume` split `ALA/GLY`; `sc_hbd`,
  `sc_polar_atoms` and `nucleophilic` split `ARG/LYS`; `beta_branched` splits
  `ILE/LEU` and is the only one of the fourteen that does.
* **Filling it is worth +0.000165, behind a control that adds the same number of
  tables from old wires.** Resolving every collision the quantiser creates buys
  nothing, so the detector's deficit is not about residue-type resolution.

That closes per-residue-type dictionaries for the detector, by two independent
routes: the representation argument and the measurement. Expanding a chemical
dictionary remains useful for candidate triage in the sibling repository, where
the quantity is read directly rather than through a quantiser. It is not a route
to pLM-NN.

**What the rule admits.** Reduce a residue to the centroid of its heavy atoms —
which is what `c_i` is, in the appendix's own conventions — and the backbone stops
existing. There is no φ, no ψ, no backbone hydrogen bond, no secondary structure,
and no Cα→Cβ direction anywhere in the 35 algebraic quantities; `phi` in
`geometric_foundation.py` is a spherical coordinate for probe directions and the
`NEAR_LAG = 4` comment in `chain_operator_descriptors.py` is a centroid distance at
sequence lag four, which is a helix proxy and not a torsion. Two different
backbone conformations can present the same centroid set, because recovering a
torsion needs N and C positions that the pipeline discards at parse time. So
backbone geometry is the first proposed family that the rule above does not
already refuse, and `pdb_io.parse_pdb_atoms` keeps the atom names, so the bytes
are on disk.

It is also the axis with a mechanism for the pLM-NN deficit rather than a hope:
the best-replicated probing result about protein language models is that they
encode secondary structure strongly, cryptic pockets open by backbone motion, and
the residues lining them sit disproportionately in loops and at helix termini.
That is a hypothesis with a named observable, which is the standard §2g failed and
§2h was measured against — so it gets a control arm that destroys the backbone
relation while preserving every marginal, not just a bigger bank.

## 2j. The seventh family is the first that is not null, and the rule predicted it

`BACKBONE_WIRES_LIFT.json` and `BACKBONE_PERMUTED_LIFT.json`, 12 cluster-disjoint
halvings, the same protocol every other family was measured on.

§2i said the deciding question is whether a family reads bytes the pipeline throws
away. `src/pocket_bench/methods/backbone_geometry.py` is the first that does: N,
CA, C, O and CB, from which it computes thirteen quantities the centroid
representation cannot express — φ and ψ as points on the circle, the Ramachandran
cell as a combinatorial label, the CA-trace turn and torsion, backbone hydrogen
bonds donated and accepted, the sequence lag to the donated partner, the CA→CB
direction against the outward radial, and CA packing with side chains removed.
Aggregated own / contact / walk2 exactly as chemistry 42 was, giving 39 columns
against its 42, so the two are a comparison of quantities and not of attachments.

| comparison | mean | 95% CI | splits |
|---|---|---|---|
| backbone − deployed | **+0.00196** | [+0.00092, +0.00300] | 8/12 |
| backbone − `more_old` control | **+0.00147** | [+0.00016, +0.00277] | 8/12 |
| **backbone − row-permuted backbone** | **+0.00267** | [+0.00140, +0.00394] | **10/12, p=0.019** |
| row-permuted backbone − deployed | −0.00071 | [−0.00170, +0.00029] | 5/12 |

**The third row is the result and the fourth is why.** The permuted arm is the
same thirteen quantities, aggregated identically, with rows shuffled inside each
chain under a fixed seed: every column's multiset of values over the chain is
unchanged, so every marginal is identical by construction and the only thing
destroyed is which residue each row describes. Permuted, the family is worth
nothing and slightly less than nothing. Unpermuted it is worth +0.002. So the
gain is the backbone conformation *of the residue being scored*, not the shape of
a column, not the extra cells, and not anything a bigger bank would supply.

Compare chemistry 42, whose union arm gave +0.000165 with its own `more_old`
control ahead at +0.000430. Backbone is twelve times the lift and reverses the
sign against the control.

**Four limits, and none of them is optional.**

* **It is small.** +0.002 against a 0.0243 deficit to pLM-NN. Even added to §2f's
  gate radius it is under a fifth of the gap. This does not beat pLM-NN and
  saying it does would be the manoeuvre this repository has already retracted
  once.
* **It sits on the reseed floor.** The decisive +0.00267 clears 0.0026 by
  0.00007. The floor is about pairing-seed noise on a different axis, so the two
  are not the same quantity, but a margin the size of the noise elsewhere is a
  coincidence to state rather than to argue past.
* **The sign test is weaker than the interval.** 8/12 at p=0.19 for the headline
  arm; only the permutation comparison reaches 10/12 at p=0.019. The intervals
  are paired over splits and the split-to-split correlation is high, which is
  what makes them narrower than the sign test is strong.
* **Adopting it is not free.** The union attachment adds 304 tables to 5152, and
  moving the architecture makes `EXTERNAL_READ.json`'s +0.0443 a result about a
  detector that no longer exists. Set B and Set C are frozen and unread for
  exactly this.

**What was actually verified, because two of the three quantities were wrong
first.** The dihedral's sine had the wrong sign, which exchanged the two helical
cells and put 127 of one 254-residue chain's residues in the left-handed cell; a
protein has a few percent. The hydrogen-bond direction test was inverted — it
asked for the nitrogen on the carbonyl carbon's side of the oxygen rather than
beyond it — and the modal donated lag through a helix came back as 2 where an α
turn fixes 4. Neither raised. Both were caught by asking for a number whose value
is known before the module runs, and `tests/test_backbone_geometry.py` now builds
an ideal backbone forwards from bond lengths, angles and torsions and asks the
module to read φ = −57° and ψ = −47° back. On real chains the CA turn comes out
at 91.7° against a textbook 89° and the CA torsion at +50.4° against +50°.

Alignment was the other place this could have failed silently. The wide cache's
rows are the sorted set of *integer* resseq, while a torsion is a property of the
polymer and needs the insertion code too; 1dc6_A has 330 polymer residues and 329
cache rows. The builder derives both lists, maps between them, and checks its own
recomputed centroids against the cache's row by row: **agreement is exact,
0.00e+00, on all 234,838 rows.** Without that check every backbone quantity past
the first insertion code would have been attached to the wrong residue and
nothing would have raised.

## 2j-bis. Expanding the backbone family doubled it, and the permutation went negative

Thirteen quantities was the conservative version. Forty-four is the same axis
taken seriously, and the answer scales with it.

`BACKBONE_WIDE_LIFT.json` and `BACKBONE_WIDE_PERMUTED_LIFT.json`, same twelve
halvings, same attachment, 132 columns against the earlier 39. The expansion is
**strictly additive**: the first thirteen quantities produce bit-identical
columns before and after, checked against the retained old cache, so §2j remains
a statement about a subset of this family rather than about a different one.

| comparison | mean | 95% CI | splits | sign p |
|---|---|---|---|---|
| backbone 132 − deployed | **+0.00441** | [+0.00331, +0.00551] | **12/12** | 2.0e-04 |
| **backbone 132 − row-permuted** | **+0.00654** | [+0.00506, +0.00801] | **12/12** | 2.4e-04 |
| row-permuted − deployed | **−0.00213** | [−0.00365, −0.00061] | 3/12 | — |
| `more_old` control − deployed | −0.00014 | [−0.00131, +0.00103] | 6/12 | — |

Three things worth keeping.

**The permutation now goes negative.** At 39 columns it was −0.0007 and crossed
zero; at 132 it is −0.0021 and does not. Columns of this shape, attached to the
wrong residue, are *worse than nothing* — they cost cells and carry noise. That
is the sharpest available statement that the +0.0044 is the backbone
conformation of the residue being scored and not the shape of a column.

**Adding tables is still worth nothing.** `more_old` adds the same 1,056 tables
from the deployed wires and lands at −0.00014 crossing zero, on the same splits
where the family gains 0.0044. Cell budget is not the mechanism.

**The axis scales.** Thirteen quantities gave +0.00196; forty-four give +0.00441.
Two points do not make a law and §2c's table is the standing warning about
reading a trend off two measurements, but the axis is plainly not exhausted, and
the expansion cost one afternoon of reasoning and 26 seconds of rebuild.

**Where this leaves the deficit, stated without decoration.** pLM-NN is ahead by
0.0243 on the official fold and 0.0340 externally. The gate radius is worth
+0.0025 and this is worth +0.0044; if they add, which is unmeasured, that is
+0.0069, or **28% of the official-fold deficit**. It is the largest honest number
this line of work has produced and it is not a win. Nothing here has been
deployed, and deploying either makes `EXTERNAL_READ.json`'s +0.0443 a result
about a detector that no longer exists.

**What is on the same rule and not yet built.** §2i's question is whether a family
reads bytes the pipeline throws away. Side-chain *conformation* does: the centroid
of a residue's heavy atoms does not determine χ₁, so a leucine in gauche⁺ and one
in trans are different geometry with identical identity, and rotamer strain is a
recognised signature of a site that opens. That family is torsions, rotamer bins
as combinatorial labels, strain against the nearest canonical rotamer, the
direction and burial of the terminal atom, an identity-normalised extension
ratio, and atom-level rather than residue-level packing counts. None of those is
a function of residue type, which is the test §2i imposes and the test chemistry
42 failed.

## 2l. Units both published baselines rank at chance, and the mirror count

`RECOVERED_UNITS_TRAIN.json`, 769 training units, no read of the held-out fold.
The request was "find a case the baselines missed and we found", and searching
769 units for one is also the most reliable way to manufacture it. Two things
make this a measurement instead: the rule is fixed in the tool before the counts
are read, and **the mirror set is reported beside it** — the units where both
baselines rank the cryptic residues well and we are at chance.

Rule: a **recovery** is `ours >= 0.80` with `p2rank <= 0.55` and
`pocketminer <= 0.55`, per-unit ROC-AUC, chance 0.50. The **mirror** swaps the
roles. Two thresholds rather than one, because a unit where everybody scores 0.7
is not a disagreement.

| found ≥ | missed ≤ | ours | mirror | difference |
|---|---|---|---|---|
| 0.80 | 0.55 | 5 | **0** | +5 |
| 0.80 | 0.60 | 8 | **0** | +8 |
| 0.75 | 0.60 | 12 | 2 | +10 |
| 0.75 | 0.65 | 17 | 7 | +10 |
| 0.70 | 0.60 | 16 | 5 | +11 |
| 0.70 | 0.65 | 24 | 11 | +13 |
| 0.65 | 0.60 | 23 | 11 | +12 |
| 0.65 | 0.65 | 31 | 19 | +12 |

**The whole ladder is reported because reporting one row of it would be a
search.** The asymmetry is positive at every setting and the mirror is exactly
zero at the two strictest, which is the strongest form the statement takes.

**The four clean cases**, after removing one for a reason given below:

| unit | residues | cryptic | ours | P2Rank | PocketMiner |
|---|---|---|---|---|---|
| `2xur_B` | 364 | 4 | 0.952 | 0.536 | 0.515 |
| `8bdi_I` | 141 | 3 | 0.890 | 0.335 | 0.408 |
| `5f3k_B` | 203 | 15 | 0.828 | 0.542 | 0.532 |
| `8c3u_A` | 152 | 13 | 0.814 | 0.409 | 0.516 |

**`4m7p_A` stays out of the joint recovery table; against P2Rank alone it is a
case study.** The deposit is an ensemble refinement: 60,040 ATOM lines, twenty
alternate conformers `A`..`T` of 3,002 atoms each. Not every published baseline
parses that file the same way, so a multi-baseline "recovery" on it is a win
about parsing — the tool still parks any unit failing that check in its own
list. **P2Rank does share the residue universe:** both sides emit scores on the
same 390 residues (`n_cryptic=29`), and the within-unit ROC-AUCs are **0.918
(counting field) against 0.439 (P2Rank)**. Report that pairwise margin only as
training-fold exploratory text; do not fold it into the four clean recoveries,
do not imply a second confirmatory SOTA miss, and do not use it to soften read
13 (0 recoveries on the official fold).

**Two of the four have very few cryptic residues.** `2xur_B` has four and
`8bdi_I` three, which is §2e's small-`n1` regime where the null sampling error of
a per-unit AUC is about 0.11. `8c3u_A` and `5f3k_B`, at 13 and 15, are the two
that do not lean on it. Say which is which when these are shown.

**It does not replicate on the held-out fold, and that is read 13.**
`RECOVERY_READ.json`, exploratory, under a plan that hashes the training-fold
tool and artifact and refuses if either moved. Same rule, 192 official units, all
three baselines this time: **0 recoveries, 1 mirror**, and the difference is
negative at all eight ladder settings (0 against 1, 1, 1, 2, 2, 3, 4, 6).

Two readings, and both belong in any sentence about it:

* **The fold has almost no power here.** 4 in 769 is one per 192, so the expected
  count on this fold is 1.0 and 0-against-1 is what that rate gives by chance
  either way. The third baseline also makes the bar strictly harder: P2Rank is
  below 0.55 on 17 units, pLM-NN on 25, PocketMiner on 26, **all three at once on
  6**.
* **On those six we are worse, and that is not a power problem.** Ahead on 1 of
  6, mean 0.297 against a best-baseline mean of 0.446. `7e5q_B` 0.588 vs 0.532,
  then `1rtc_A` 0.522, `4gv9_A` 0.272, `1vsn_A` 0.150, `7f2m_B` 0.137, `3ly8_A`
  0.110. **The chains all three published methods fail on are chains this
  detector fails on harder.**

So the training-fold table is not evidence that the field sees sites the
baselines cannot. Both results are in the README, because showing only the first
is the selection the preregistration exists to prevent, and the sentence
reporting the negative was fixed in the plan before the read.

**What this is not.** pLM-NN is not in the training-fold counts. It has never been run on the training
fold — that is a five-hour encoder pass, not a lookup — so every count above is
against **two** baselines. Any sentence about "P2Rank and pLM both missed it"
needs the official fold, where all four methods' per-residue scores are already
frozen, and that is a read of the held-out set: it needs a plan and an index
before it is run, and the rule must be the one fixed here rather than one chosen
after looking.

## 2k. The ledger was under-counting, and a gate that greps one spelling grades spelling

Not a measurement of the method — a measurement of the accounting, and it found
a hole in the thing the paper's honesty section is built on.

**What was found.** `results/architecture_sweep/` is the training-fold directory,
and `AGENTS.md` requires every training-fold artifact to say
`"reads_test_fold": false` and mean it. Twenty-seven artifacts there said nothing
at all, the whole `COUNTERATTACK_*` series among them, and three also lacked
`clinical_grade`. Deriving the declaration from each artifact's generator rather
than stamping it — the generator is located by requiring that it *binds the path
and writes through it*, since `emit_frozen_numbers.py` binds nearly every path in
the repository and counting readers as writers made seventeen artifacts look
guilty — gave twenty that no generator could have read the fold in, and **seven
that did**.

**Those seven were invisible to the test-fold access ledger.** Its rule was: a
per-unit table of at least 150 rows keyed `unit_id` with a column containing
`auc`, or a declared `test_fold_read_index`. Two ways through it:

* `FULL_EXPANSION.json` — `run_full_expansion.py` loads `_cascade_cache_test.npz`,
  takes `yte`, and reports twelve ROC-AUCs over `n_test_units: 192`. It stores
  only aggregates, so there is no table to find.
* `DUAL_TRACK_AB.json` — carries 192 per-unit ROC-AUCs and still escapes, because
  the row key is `unit` not `unit_id` and the metric is `A_resolved` not anything
  containing `auc`. **Every one of its `unit` values is `null`**, so it scored the
  held-out fold without recording which units it scored.

A third signal now catches both: an artifact reporting a metric while naming the
fold's unit count has taken a number off it, in whatever shape. Standalone probes
**13 → 20**; six of them scored the fold. `n_distinct_architectures_evaluated`
stays at **12**, because all seven are table-field variants already represented,
so the figure caption and the selection-bias arithmetic in the paper are
unchanged. Only `\NStandaloneProbes` moved.

**The generalisable part.** The tree spells "did this touch a held-out set"
**eighteen** different ways, six of which answer this exact question:
`reads_test_fold`, `test_fold_touched`, `test_fold_read`, `test_fold_reads`,
`reads_our_test_fold`, `reads_cryptobench_test_fold`. `GAP_DECOMPOSITION.json`
was honest — it says `test_fold_reads: 0` — and looked silent to a gate that knew
one spelling.

> **A gate that greps for one field name measures the field name.** Before
> trusting a gate that reports zero, enumerate the spellings actually present in
> the tree and check that the gate's is the one they use. `AGENTS.md` already
> says a query returning zero has not told you there is nothing there; this is
> that rule applied to our own gates rather than to RCSB.

`tools/train_fold_declarations.py` holds the synonym list, derives the
declaration from the generator, takes the *direction* from the ledger so a file
the ledger records as an access can never be stamped `false`, and runs in
`make verify`. The audit is closed: 27 of 27 declared, 7 of them `true`.

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

**Integer graph invariants of the contact lining — built, screened, being measured.**
Fifteen invariants of the 7 Å pinch lining that G2's `betti0` and `betti1` use:
triangles through the residue and in the wall, four-cycles, k-core number,
eccentricity, largest clique containing it, induced P4s, two-step expansion, and of
the wall alone its girth, diameter, Wiener index, bridges, articulation points, leaves
and bipartiteness. All counts. Built to the §2c rule — fifteen different invariants of
one graph, not one invariant of fifteen graphs — and each at one radius, because
sweeping a radius is the harmful category. 87 s to build over 234,838 residues;
expanded through the fifteen deployed statistics to 225 wires because 15 columns make
only ~112 tables, too small a block for a null to mean anything.

The screen put it at **+6.42e-06**, beside composition and nowhere near the deployed
bank, so the prediction committed before the run was **no material field lift, between
−0.002 and +0.001**, with a lift outside that range falsifying the screen.

**Measured, and it is closed as an axis and as a screen** (`APPENDED_FAMILY_LIFT.json`,
12 splits): field **−0.0061 on 0/12** under the union attachment and −0.0064 widened,
solve **−0.0027 on 1/12**. The first split's −0.0091 held up. Both readouts lose, so
the fifteen invariants are not a family the bank was missing — see §2c for what that
does to the screen, which is that it does not survive it. The build rule the family
was constructed under, fifteen different invariants of one graph rather than one
invariant swept, is *not* what failed here: that rule came from the same-quantity
decomposition, which is a separate measurement and still stands. What failed is the
belief that an internal interaction statistic can tell you in advance whether a family
is worth attaching.

The one quantity that was never computed, and the only remaining candidate
explanation, is the **cross interaction between the new columns and the deployed 645
wires**. A family can have positive internal interaction and still be redundant with
the bus, and the union attachment forms no straddling tables at all, so redundancy
arrives as tables whose content is already present for the fan-out to decorrelate
away. **Now measured, and it points at the attachment rather than at the columns —
see §2d.**

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

**A run that dies with no traceback was killed, and the first suspect is the machine
and not your code.** `collectability_screen.py --cross` was launched three times and
each time the log stopped after the family-build line with no exception, no exit
status and no artifact. Nothing was wrong with the tool. The host is 16 GB; Cursor
was holding 6.3 GB in one renderer, Chrome 3.2 GB, total RSS 14.6 GB and swap 7.2 of
8 GB, leaving about 250 MB. The kernel takes whatever grows into that and takes it
with `SIGKILL`, which is why the log ends mid-run. Read `vm_stat` and
`sysctl vm.swapusage` before reading your own diff. Note also that the same tool's
heavier sibling, `appended_family_lift.py`, completed twelve splits on this machine
two hours earlier — the tool did not get bigger, the machine got smaller.

Two of the changes made to fit it are worth keeping whatever the memory situation,
and both are the reimplementation rule applied to arithmetic rather than to a solve:

- **Digitise once, release the float columns, and do not upcast.** The wide cache is
  234,838 × 645 float32, 578 MB, and `np.asarray(..., dtype=np.float64)` doubles it
  to 1,156 MB — the single largest allocation the tool makes. `chain_digits` ranks
  within each chain over all of that chain's residues, so its output does not depend
  on the split, and nothing downstream reads the floats. Skipping the upcast is exact,
  because float32 to float64 preserves both order and equality and so both the stable
  argsort and the tie grouping see the same thing. The tool checks that on 40 chains
  and refuses rather than assuming it.
- **Form only the block you read.** The cross term needs the straddling block of a
  Gram over 645 + n columns. Building the whole joint Gram and slicing it allocates
  two `(645+n)² × 16` tensors for an answer that is a tenth of one: about 1.1 GB of
  transient for 18 MB of result. The rectangular product is formed directly and is
  required to reproduce `joint_counts` on a 24 × 12 subset.

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

**Stage by path, and `git add -A` failed again.** A commit that froze Set B swept in
a sweep's artifact because it was staged with `git add -A` while that run had just
finished writing. The artifact landed in the freeze commit and its analysis in the
next one, which is the wrong way round and makes both diffs harder to read. `AGENTS.md`
already has this rule from a 292 MB incident; this is the second recorded instance and
the failure mode is background runs finishing between staging and committing.

**Measure a solve against a solve.** A tool built to compare what a counting field
and a linear solve each take from a new column family first reported the solve arm
against the frozen *counting field*, so its −0.019 was mostly the two readouts
differing — which on the deployed wires is +0.0053 — and not anything the family
contributed. The lift of a readout has to be measured against the same readout
without the addition: `fisher(old + new) − fisher(old)`, never `fisher(old + new) −
field(old)`. Caught in a smoke test because the number was implausibly large.

**A field can exist and still not mean what you assume.** A diagnostic summed pair
verdicts by walking a set's units and reading `u["pairs"]`. The builder strips
`pairs` from `units_without_a_cryptic_pocket`, so it counted only the pairs of units
that ended up with a pocket and reported Set B as 84 % cryptic. The right source was
`pair_verdicts`, which the builder computes over every pair it examined. The tell was
that 84 % is absurd; the lesson is that a present field can be a narrower population
than its name suggests, and the tool now refuses rather than falling back.

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

**Set B and Set C are built and frozen, and neither has been read.** Both are
single-particle cryo-EM past CryptoBench's cutoff, cluster-disjoint from CryptoBench,
from Set A and from each other, with Set A's labelling machinery imported unedited so
a read on either is comparable to the read on Set A.

| set | ceiling | candidates | units | residues | digest |
|---|---|---|---|---|---|
| Set B | 2.5 Å | 82 | **8** | 143 | `09381b40…` |
| Set C | 3.0 Å | 217 | **38** | 584 | `ff112a60…` |

They were frozen *before* any method change, which is the whole point: because they
are unread, a read at any future time is still valid. Do not read either to see how
it goes. A read spends it, and with no improvement to confirm it buys nothing —
8 units would give an interval far too wide to conclude from.

Two counts on the way there were wrong and both were caught by running the real
selection instead of extrapolating a pool number. `SETB_POOL.json`'s 455 accessions
"with apo and holo" counts *any* ligand on the chain; Set A's rule wants one of
CryptoBench's 2,404 accepted codes, which is 102 on the same inventory — the loose
rule over-counts by 4.5×. And the first strict counts came back 40 at 2.5 Å and 46 at
3.0 Å, whose near-equality was the tell: the cached UniRef50 map covered only the
2.5 Å pool, so 281 accessions were dropped as "UniProt cannot cluster it" when they
were merely absent from the map.

**A cryo-EM label is not resolution-neutral, and the direction is knowable.** Pairs
the recovered rule declines to label run 9.2 % for X-ray Set A, 17.3 % for Set B and
31.9 % for Set C — a third of Set C cannot be decided. The cryptic share rises with
it, 6.0 → 7.1 → 12.2 %. Over 11,437 pairs of units with different labels the coarser
one is the cryptic one 56.0 % of the time against 50 for no relation, and the finest
25 units yield no cryptic call at all. That is real, weak and not monotone, so it does
not establish that Set C's higher yield is coordinate noise inflating pRMSD — but
resolution is a covariate of the label and belongs in front of anyone reading either
set (`CRYOEM_LABEL_SENSITIVITY.json`).

The order is not negotiable and has been followed: build, freeze, hash, *then*
preregister, *then* read once. No preregistration exists for either set yet, which is
correct while there is nothing to confirm.

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

**A protein language model is not covered by that allowance, in either
direction.** The ridge solve is allowed as a screen because it is arithmetic
this repository performs on its own inputs. An encoder trained elsewhere on
millions of sequences is not: reading it as a diagnostic, quantising it into
wires to see whether the pLM-NN deficit is information or readout, distilling it
into anything — all of it is out, and the reason is not purity for its own sake.
A number produced with an encoder in the loop cannot be checked by someone who
does not trust us without them obtaining the same 5.7 GB of weights, which is
the property the whole repository exists to have.

`pLM-NN` stays as a **baseline**: CryptoBench published it, it beats us by
0.0243 on the official fold and 0.0340 externally, and reporting a method that
wins is the honest thing to do. It is rebuilt under `tools/plmnn_*.py`, it never
touches `src/`, and since this was previously true-but-unchecked there is now a
gate. `detector_reads_no_learned_model` in `tools/verify_claims.py` walks every
module under `src/` and fails on an import of a learning framework or a
reference to the cached encoder's artifacts; `tests/test_detector_purity.py`
holds it, including a case asserting the gate still catches a violation and one
asserting that `sequence_wires.py` may keep *saying* ESM-2 is excluded. The rule
matches artifact names rather than the model family precisely so that
documenting the exclusion stays legal and performing a read does not.

The temptation this guards against is specific and will recur: the encoder is
already in the iCloud cache, the deficit is the only open number, and every
session that reaches for it will find the shortest path runs through weights
somebody else fitted. Take the long way or leave the number where it is.

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
| `geoaudit-cryptobench` | this one; the benchmark paper | 27 gates, 732 tests, in sync |
| `foliation-transfer-atlas` | zero-tuning transfer to 5 oncology targets | committed and pushed; no transfer run yet |

The transfer atlas holds its own `AGENTS.md`, twelve gates and a three-grade
exposure design. Its next measurement is already named: for menin's `4GQ3`
pocket, AHoJ-DB records `pocket_rms` between 0.51 and 1.16 across all ten apo
chains, every one under the 2.0 Å cryptic threshold, so that pocket carries no
cryptic label and whether menin has one at all is an open question about the
target rather than about the implementation.
