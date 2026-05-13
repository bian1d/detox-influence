# RLHF Influence Function Project — Agent Instructions

## What this project is

This is the implementation of a master's thesis experiment by a student
(Yide) on tracing PPO/RLHF rollout-level influence in a detoxification
setting. The thesis investigates which training rollouts most influenced
the PPO-trained policy's detoxification behavior, using EK-FAC influence
functions over a chosen MLP layer's W_2 matrix.

The full theoretical background is in `CONTEXT.md`. Read it before doing
anything.

## Update log

This project evolved through three phases of decisions. The current state
reflects empirically grounded choices, some of which differ from the
original plan:

- **2024-XX → Model pivoted from GPT2-medium to GPT-Neo-125M.** After 7
  rounds of PPO debugging on GPT2-medium revealed structural k1 KL
  controller instability (controller collapse + KL going deeply negative
  during detox), we switched to GPT-Neo-125M. This is the model
  ybelkada/gpt-neo-125m-detox uses in the TRL official detox tutorial,
  and it trains stably with TRL's default settings.

- **Target layer adjusted: layer 9 of GPT-Neo (not layer 19 of GPT2-medium).**
  Lee et al. identified layer 19 in GPT2-medium (24 layers, ≈79% depth)
  as containing the most toxic value vector. We map this proportionally
  to GPT-Neo-125M's 12 layers: layer 9 ≈ 75% depth. This is the
  primary; layer 10 is a secondary candidate. Layer name in code:
  `transformer.h.9.mlp.c_proj`.

- **θ\* checkpoint: step_0650 (not the final step_1999).** Phase 0
  three-phase training analysis identified step_0650 as Pareto-optimal
  across toxicity (0.016), reward (3.84), PPL (33), and length (10).
  Phase A (0-300) was rapid detox learning; Phase B (300-1200) was
  healthy plateau with peak at step 650; Phase C (1200-2000) was
  over-training with reward hacking onset, as independently confirmed
  by wandb-logged `ppo/val/var_explained` dropping below 0.5 sustained
  after step 1165.

- **EK-FAC validation gate revised.** The original 1e-3 Frobenius gate
  against brute-force F⁻¹ proved structurally unattainable on
  transformer architectures (~45% off-diagonal Kronecker mass; this is
  architectural, not implementation error). Replaced by 4 machine-
  precision implementation gates + at-scale self-influence sanity. See
  Hard Constraint 5 below.

- **Eval target f: hybrid (f_seq from Hu et al. 2025 + f_toxic).**
  Hu et al. 2025 ("A Snapshot of Influence", NeurIPS 2025 Oral)
  demonstrated f_seq (sequence-level with last-token advantage) is more
  effective than naive f_return for RLHF attribution. We adopt f_seq as
  primary production target and a hand-crafted f_toxic as secondary
  target for Phase 3 mechanistic anchoring.

- **PPO generation parameter: min_new_tokens=10 (not min_length=10).**
  HuggingFace `min_length` is total sequence length (prompt + completion),
  not new tokens. Run 1 collapsed at step 902 with min_length=-1. Run 2
  collapsed at step 1012 with min_length=10 (long prompts made constraint
  vacuous). Run 3 succeeded with min_new_tokens=10 which directly masks
  EOS logit for first 10 generated tokens regardless of prompt length.

## Repository layout

```
.
├── CLAUDE.md                  ← you are reading this
├── CONTEXT.md                 ← theoretical background
├── requirements.txt
├── phases/                    ← per-phase task documents
│   ├── phase0.md
│   ├── phase1.md
│   └── phase2.md
├── docs/                      ← reference papers (read-only)
│   ├── DPO-detoxify-paper.pdf
│   ├── MDA-paper.pdf
│   └── detoxify-trl-official.md
├── references/                ← reference code (read-only, do NOT import)
│   ├── DPO-detoxify/          ← Lee et al.'s probe + DPO code
│   └── MDA-EK-FAC/            ← Chen et al.'s EK-FAC implementation
├── src/                       ← all code under here
│   ├── train_ppo.py
│   ├── train_probe.py
│   ├── verify_probe.py
│   ├── eval_toxicity.py
│   ├── eval_ppl.py
│   ├── evaluate_checkpoints.py
│   ├── ekfac/                 ← Phase 2's EK-FAC pipeline
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── data.py
│   │   ├── toy.py
│   │   ├── hooks.py
│   │   ├── factors.py
│   │   ├── eigen.py
│   │   └── influence.py
│   └── phase1/                ← Phase 1's variance diagnostic (pending)
├── tests/                     ← pytest tests + diagnostic scripts
│   ├── test_eigen_real.py, test_eigen_toy.py
│   ├── test_factors_real.py, test_factors_toy.py
│   ├── test_influence_toy.py, test_self_influence_real.py
│   ├── test_mask.py, test_toy.py, test_training_toy.py
│   ├── test_eval_ppl.py, test_eval_toxicity.py, test_verify_probe.py
│   └── diag_*.py              ← debug-time diagnostic scripts (kept for
│                                 reference, not run by pytest)
├── data/                      ← all data artifacts (gitignored)
│   ├── archive/               ← failed PPO runs preserved for reference
│   ├── jigsaw/                ← Jigsaw probe training data
│   ├── probe/probe.pt         ← our trained probe on GPT-Neo-125M
│   ├── eval_prompts_400.json  ← Phase 0 fixed eval prompt set
│   ├── baseline_ppl.json, baseline_toxicity.json
│   ├── ppo_checkpoints/       ← step_0000 through step_1999
│   ├── rollouts/              ← step_NNNN.pt files, 651 used by Phase 2
│   ├── smoke/                 ← PPO smoke test artifacts
│   ├── ppo_train_log.jsonl    ← per-step training metrics
│   ├── checkpoint_eval.csv    ← Phase 0 evaluator output
│   ├── ekfac/                 ← Phase 2 intermediate (created during run)
│   └── *.log                  ← run logs (ppo_full_run, eval_smoke, etc.)
├── reports/
│   ├── phase0_report.md
│   ├── phase2_report.md
│   ├── phase0_checkpoint_curves.png
│   └── notes/                 ← debug notes, research observations
│       ├── kfac_independence_mismatch.md
│       ├── overparameterized_loo_breakdown.md
│       ├── stage3_loo_report.md
│       ├── stage3_real_self_if_report.md
│       ├── loo_investigation.md
│       └── step_277_anomaly.md
└── wandb/                     ← W&B run artifacts (gitignored)
```

All new code goes under `src/`. Tests go under `tests/`. Data and
artifacts go under `data/`. Reports go under `reports/`. Never write
into `references/` or `docs/` — those are read-only.

## Reference materials

The `references/` directory contains read-only copies of prior projects
we draw on:

- `references/DPO-detoxify/` — Lee et al.'s DPO toxicity paper code
  (https://github.com/ajyl/dpo_toxic). Use this to understand how
  `probe.pt` was constructed for GPT2-medium and as a positive control
  for probe diagnostics. The repo does NOT contain probe training code
  (probe.pt is committed as pre-computed artifact); the paper is the
  reference for methodology.

- `references/MDA-EK-FAC/` — Chen et al.'s MDA EK-FAC reference
  implementation. Used as structural reference for Phase 2's EK-FAC
  pipeline (Stage 1A factor accumulation, Stage 1B eigendecomposition,
  IHVP, per-sample gradient scoring). Note: their setting is SFT not
  PPO, so adaptation is needed.

**Hard rule: do not `import` from `references/`.** Reimplement what
you need under `src/`. Reasons:
1. The reference projects have their own dependencies and abstractions
   that do not match ours.
2. Direct imports create silent coupling.
3. Reimplementing forces actual understanding, which is needed for
   thesis writing.

The corresponding papers are in `docs/`:
- `docs/DPO-detoxify-paper.pdf` — Lee et al. 2024, "A Mechanistic
  Understanding of Alignment Algorithms: A Case Study on DPO and
  Toxicity"
- `docs/MDA-paper.pdf` — Chen et al., "Mechanistic Data Attribution"
- `docs/detoxify-trl-official.md` — HuggingFace TRL detoxification
  tutorial

Additional reference cited in the project but not stored locally:
- Hu et al. 2025, "A Snapshot of Influence: A Local Data Attribution
  Framework for Online Reinforcement Learning" (NeurIPS 2025 Oral,
  arxiv 2505.19281). Most directly comparable prior work; we adopt
  their f_seq objective for our production eval target. Reference via
  arxiv, not stored in `docs/`.

## Current decisions (do not question or change)

- **Model**: GPT-Neo-125M (12 layers, hidden_size=768, intermediate_size=3072)
- **RL algorithm**: PPO via HuggingFace TRL 0.9.6
- **Reward model**: facebook/roberta-hate-speech-dynabench-r4-target
- **PPO config**: `lr=2.94e-5, batch_size=32, init_kl_coef=0.2,
  adap_kl_ctrl=True, target=6.0, kl_penalty='kl', min_new_tokens=10`
- **Parameter subspace φ**: W_2 of layer 9 MLP
  (`transformer.h.9.mlp.c_proj`, shape `(3072, 768)`). Layer 10 is
  secondary candidate.
- **θ\* checkpoint**: `data/ppo_checkpoints/step_0650/`
- **Production rollouts span**: `data/rollouts/step_0000.pt` through
  `data/rollouts/step_0650.pt` (~20,832 rollouts, ~600k tokens)
- **Eval target f**: hybrid (f_seq primary, f_toxic secondary)
  - f_seq: `E_{x~in_training, y~π_θ*}[log π_θ(y|x) · advantage_{-1}^θ*]`
    following Hu et al. 2025 §5.3
  - f_toxic: `log π_θ(y_toxic | x_eval)` for hand-crafted (x_eval, y_toxic)
    used for Phase 3 mechanistic anchoring
- **KL estimator**: k1 (TRL default). All KL-using computations use k1
  consistently across Phases 0, 1, 2.
- **Influence formula**: I = -g_eval^T F^{-1} s_m (KL-RL convention).
  For supervised toy validation, +g_eval^T F^{-1} s_m (sign flipped).
  Implementations: `influence_score_klrl` and `influence_score_supervised`.
- **Fisher approximation**: EK-FAC with two-level damping
  `max(Λ + 0.1·Λ̄, 1e-5)`.

## Phase structure

- **Phase 0** ✓ closed: Environment setup, baselines, PPO detoxification
  training with dense checkpointing. Three runs documented (two with
  failure modes preserved as engineering findings). 64,000 rollouts
  produced.
- **Phase 1**: Effective reward variance diagnostic on θ* = step_0650.
  Validates IF formula's stationarity assumption is met in our KL-RL
  setting.
- **Phase 2**: EK-FAC influence function implementation. Implementation
  correctness verified at machine precision. At-scale self-influence
  sanity passed (5/5 sign-consistent, top-5 magnitude). Awaiting
  production IF run on θ* against f_seq + f_toxic.
- **Phase 3**: Mechanistic verification — connect top-IF rollouts to
  toxic neurons. Layer selection finalized post Phase 2. Probe
  reproduction on GPT2-medium as ablation study for whether GPT-Neo's
  toxic mechanism is distributed (Yang et al. 2025 follow-up).

## Hard engineering constraints

These are non-negotiable. Violating any of them will silently produce
wrong numbers.

1. **Prompt mask is mandatory.** Every loss computation
   (Stage 1A/1B/3) must mask out prompt tokens with `ignore_index=-100`.
   The prompt is not generated by π_θ and must not enter
   log π_θ(y|x).

2. **Stage 3 must be batch_size=1.** IF is a per-sample quantity.
   `autograd.grad` on a batch returns Σ_i ∇L_i, which cannot be
   decomposed into per-sample gradients without `torch.func.vmap` or
   functorch. If unsure, use batch=1 with a Python loop.

3. **Reduction is `mean`, not `sum`.** PPO rollouts have variable
   length; `sum` reduction biases longer rollouts to higher influence
   scores in a way Fisher inverse cannot fully correct. Both Stage 1
   accumulation and Stage 3 must use `mean` (per-token average) and
   they must be consistent.

   Note: PPO reward shaping in `train_ppo.py` may use different
   reduction conventions per TRL standard practice. The `mean` rule
   here applies specifically to IF computation (Stages 1A, 1B, 3),
   not to the PPO training objective.

4. **Stage 1 uses pseudo labels, Stage 3 uses recorded labels.** Stage
   1 estimates Fisher (which requires E_{y~π_θ}); pseudo labels
   sampled from π_θ are correct here. Stage 3 measures the influence
   of an actual training rollout, so use the recorded response tokens.

5. **EK-FAC validation gates** (revised based on Phase 2 empirical
   findings). The original 1e-3 Frobenius gate against brute-force F⁻¹
   proved structurally unattainable on transformer architectures due
   to K-FAC independence violation (~45% off-diagonal Kronecker mass;
   this is architectural, not implementation error). The revised
   acceptance is:

   - **Implementation correctness** (4 toy-model gates, all at machine
     precision, < 1e-10):
     - Gate 1a: Λ vs diag(Uᵀ F^(β) U) on toy model
     - Gate 1b: synthetic-independence reconstruction — under iid
       m ~ N(0, A_pop), δ ~ N(0, S_pop), EK-FAC recovers S_pop ⊗ A_pop
     - Gate 1c: Q orthonormality — ||Q_A Q_Aᵀ - I||, ||Q_S Q_Sᵀ - I||
     - Gate 1d: A, S reconstruction — A ≈ Q_A diag(Λ_A) Q_Aᵀ

   - **At-scale validation**: real-data self-influence sanity must show
     5/5 sign-consistent self-IF results, with |I_self| ranking in
     top-5 across all rollouts in the test file.

   - **LOO retraining-based validation is acknowledged as infeasible**
     for transformer settings per Bae et al. 2022 (Proximal Bregman
     Response Functions). Empirically confirmed in Phase 2 by
     brute-force F⁻¹ control experiment, which shows the LOO-IF
     discrepancy is structural to the IF formula in non-stationary
     overparameterized regimes, not implementation error.

   - **EK-FAC inverse Frobenius rel err vs (β) Fisher**: report as
     informational (~8% on transformer toy, within Grosse 2023 / MDA
     published 5-10% norms). NOT an acceptance gate.

6. **No batched autograd.grad in Stage 3.** Same idea as constraint 2
   but worth restating: if you find yourself stacking multiple
   rollouts into one tensor and calling `.backward()` once, you have
   made a fundamental error. One rollout, one backward pass, one score.

7. **Determinism for retraining-based validation.** If you implement
   any retraining-based test (LOO etc.), use `torch.manual_seed`,
   `torch.use_deterministic_algorithms(True)`,
   `torch.backends.cudnn.deterministic=True`,
   `torch.backends.cudnn.benchmark=False`, and verify bit-identical
   parameters from two runs of `train_toy_to_optimum(seed=42)` before
   running the actual LOO loop.

## How we work together

### Before writing any code on a new phase

1. Read `CLAUDE.md`, `CONTEXT.md`, and the relevant `phases/phaseN.md`.
2. Reply with:
   - The files you plan to create or modify
   - Key function signatures
   - The step you think is most likely to fail
   - What small-scale toy data you will use for initial validation
3. Wait for human review. **Do not write code yet.**

### When writing code

- Write tests **first**, especially for the IF computation. Toy-model
  numerical tests against a hand-computed answer are the strongest
  guarantee.
- Run `pytest` and confirm green before continuing.
- Commit at meaningful checkpoints with clear messages.

### When finishing a phase or stage

Write a stop-and-report message containing:

1. What was actually done (1-2 sentences)
2. List of output files (paths + sizes)
3. Each acceptance criterion from `phases/phaseN.md` with its actual
   numerical value and pass/fail
4. Any deviations from the plan, and why
5. Anything you skipped or did not finish, **explicitly listed**

Do not start the next stage until the human reviews.

### Stop-and-report protocol on acceptance failure

If an acceptance gate fails, **DO NOT** silently lower the bar.
Surface with:
- What you measured vs the threshold
- What you've already tried
- Three hypotheses for the cause
- What evidence would discriminate between them

This rule exists because Phase 0 had multiple instances of agents
silently relaxing thresholds. The user explicitly does not want this
pattern.

### Things to never do

- Do not change locked-in decisions on your own initiative. If you
  think one is wrong, stop and ask.
- Do not import from `references/`. Reimplement under `src/`.
- Do not "optimize" the implementation in ways that change semantics
  (e.g., switching reduction from mean to sum to "match standard
  practice").
- Do not skip phases or substeps to "save time." This is research
  code; correctness over speed.
- Do not silently fall back when something fails. If a step doesn't
  produce the expected output, stop and surface the problem.
- Do not paraphrase or simplify acceptance criteria.

## Code quality expectations

- Python 3.10+, PyTorch 2.x.
- Type hints on all public functions.
- Docstrings explain *why*, not *what* the code does.
- One function per concept; no 200-line god functions.
- Use `pathlib.Path`, not string paths.
- All randomness behind explicit seeds; record seeds in reports.
- Numerical work uses fp64 for accumulation (A, S, Λ), fp32 for
  per-sample gradient computation.

## Questions you should ask the human, not guess

- "What if the reward model returns NaN?"
- "The rollout is 1024 tokens, do I truncate or skip?"
- "EK-FAC eigenvalues have very small magnitudes, should I increase
  damping?"
- "The toy-model IF score doesn't match ground truth at 1e-3 but does
  at 1e-2 — should I investigate further or accept?"
- "Should θ* be step_0650 or step_1999? Which checkpoint is meant?"
- "Does this rollout file span fit memory for the EK-FAC factor
  accumulation pass?"

These are research decisions, not implementation details. Surface them.
