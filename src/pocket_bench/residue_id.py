"""One definition of which residue an identifier refers to.

Residue identity is upstream of every number this project reports: it decides
what the evaluation universe contains, which prediction lines up with which
label, and therefore what the denominator of a mean is. It used to be defined
twice -- once in the scorer and once in the P2Rank adapter -- and the two copies
disagreed, in opposite directions:

* the scorer read every digit and discarded the minus sign, so an expression-tag
  residue numbered -1 became residue 1 and the two exchanged scores. Eleven of
  the 192 official test structures carry such a tag.
* the adapter called ``int()`` on P2Rank's ``residue_label`` and skipped the row
  when that raised, so a residue with an insertion code -- '132A' -- had
  P2Rank's answer for it silently dropped.

Neither raised. Both moved published numbers. So the rule lives here, once, and
both callers import it.

**The rule.** A residue is ``(chain, resseq)``. The number is signed. The
insertion code is not part of the identity, and neither is the alternate-location
indicator; identifiers that differ only in those refer to the same residue and
their values are combined by the caller rather than one of them being dropped.

Collapsing the insertion code is forced rather than chosen. CryptoBench marks a
cryptic site with bare integers, so no label can distinguish 132 from 132A; a
finer key would add residues to the universe that no label can ever mark
positive, and every one of them would count as a true negative. One structure on
the official fold, 2v6m_D, is affected, and five residues in it share a slot.
``tools/audit_residue_identity.py`` measures that cost rather than assuming it.
"""
from __future__ import annotations

from typing import Any

__all__ = ["resseq"]


def resseq(rid: Any) -> int | None:
    """The residue number an identifier refers to, or None if it carries none.

    Accepts an ``int``, a bare number as text, and the ``'A:ALA123'`` form that
    ``pdb_io`` emits. Reads the trailing run of digits together with a minus
    sign immediately before it, which is why ``'A:GLY-1'`` gives -1 and
    ``'A:ALA123'`` gives 123. A trailing insertion code is skipped, so ``'132A'``
    gives 132 -- deliberately the same answer as ``'132'``.
    """
    if isinstance(rid, int):
        return rid
    text = str(rid)
    digits = ""
    negative = False
    for ch in reversed(text):
        if ch.isdigit():
            digits = ch + digits
        elif digits:
            # The first non-digit below the number ends it. A '-' there is the
            # sign: the identifier forms in use separate the chain with a colon,
            # so a hyphen in that position is never a delimiter.
            negative = ch == "-"
            break
    if not digits:
        return None
    return -int(digits) if negative else int(digits)
