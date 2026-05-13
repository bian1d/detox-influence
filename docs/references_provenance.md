# External References

This project draws on two third-party codebases, stored locally
but not committed to this repository (see .gitignore).

## references/DPO-detoxify/

- Source: https://github.com/ajyl/dpo_toxic
- Commit used: 34319c9bad0d22608e30460807a54869e84f474f
- Purpose: Reference for Lee et al. 2024 probe training methodology
  (Mechanistic Understanding of DPO and Toxicity, NeurIPS 2024).
  Their pre-trained probe.pt serves as a positive control for the
  Phase 0 probe reproduction diagnostic.
- License: See upstream LICENSE.

## references/MDA-EK-FAC/

- Source: Unrecoverable from local artifacts (no .git metadata, no
  upstream URL hints in code headers). Local copy timestamped
  2026-05-07. Purpose was reference for EK-FAC pipeline structural
  patterns (Chen et al., Mechanistic Data Attribution).
- Commit used: unknown — reference copy downloaded prior to project
  git tracking
- License: See upstream LICENSE if obtainable.

To reproduce this project, clone each upstream at the listed
commits into the local references/ directory. The project's own
code in src/ reimplements all needed logic and does not import
from references/ (this rule is enforced in CLAUDE.md).
