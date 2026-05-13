# Phase 2 — EK-FAC Pipeline Implementation (Kickoff)

You are starting Phase 2 of an RLHF influence-function research project. Phase 0 (PPO training) is currently running in a separate terminal — that work does not concern you. Your job is to implement the EK-FAC influence-function pipeline that will eventually consume Phase 0's rollouts.

This document gives you everything you need to start. Read it carefully, then complete the kickoff items at the bottom before writing any pipeline code.

---

## Project Context

**The user**: 5th-year PKU EECS undergrad doing thesis on rollout-level Influence Functions for PPO-trained RLHF detoxification.

**Setting**:
- Base model: `EleutherAI/gpt-neo-125m` (hidden_size=768, num_layers=12, intermediate_size=3072)
- RL framework: TRL 0.9.6 PPOTrainer
- PPO is being trained to detoxify the model on RealToxicityPrompts using a RoBERTa hate-speech reward model (running in a parallel terminal — leave it alone)
- After PPO completes, rollouts will be at `data/rollouts/step_NNNN.pt` (~32 rollouts per step × 2000 steps total)
- Each rollout file contains a list of dicts with keys: `prompt_text, prompt_token_ids, response_text, response_token_ids, policy_logprobs, ref_logprobs, reward, step`

**Your output**: A working EK-FAC pipeline in `src/ekfac/` that computes influence scores `I(z_m, f)` for each rollout `z_m` against an eval objective `f`.

**Critical: Phase 2 is independent of Phase 0's PPO completion.** You can develop and unit-test the entire pipeline against synthetic data, a toy model, AND partial rollout data (some `data/rollouts/step_NNNN.pt` files already exist as PPO runs).

---

## REQUIRED READING — Do this BEFORE writing any code

1. **Read `CONTEXT.md` in full.** It is the theoretical foundation for the entire project. You cannot implement EK-FAC correctly without understanding sections 1-8. Especially:
   - Section 1-2: KL-regularized RL objective and the optimal policy form (this is where β > 0 matters)
   - Section 4: empirical Fisher matrix definition
   - Section 5-6: the IF formula, what `φ` is, and how `f = log π(y_eval|x_eval)` plugs in
   - Section 7-8: effective reward variance (the Phase 1 diagnostic, which is downstream of you but informs why the math has to be exactly right)
   - The KL estimator note in Section 1 — Phase 0 uses k1 (TRL default). Phase 2 IF score itself doesn't involve KL estimation directly, but the parameter subspace φ choice is informed by where Phase 0's training actually shifts weights.

2. **Read `CLAUDE.md` in full.** Hard constraints (see "Hard Constraints" below for the Phase 2 subset). Permission rules. Reporting protocol.

3. **Skim `references/MDA-EK-FAC/`** — Chen et al.'s reference implementation of EK-FAC for SFT influence functions. **Read-only**. Treat as pattern reference; do not import or copy. Their setting is SFT not PPO, so direct copying would propagate setting-specific assumptions.

4. **Inventory `data/rollouts/`** to see what real rollouts exist currently. PPO is generating these in real time. Confirm the rollout schema matches what's written above.

---

## Theoretical Framework (Quick Reference Only — full derivation in CONTEXT.md)

The influence score formula:

```
I(z_m, f) = -∇_φ f(θ*)ᵀ · F_φ(θ*)⁻¹ · ∇_φ log π_θ*(y_m | x_m)
```

Where:
- `z_m = (x_m, y_m)` is one training rollout (prompt + response)
- `f` is an eval objective. For Phase 2 default: `f = log π_θ(y_eval | x_eval)` on a held-out toxic sample
- `φ` is a parameter subspace (a single MLP W_2 matrix in one layer — see Layer Selection below)
- `F_φ(θ*)` is the empirical Fisher matrix restricted to φ
- `θ*` is the post-PPO model (or any intermediate checkpoint — for development, use whichever exists)

EK-FAC is a Kronecker-factored low-rank approximation of `F_φ⁻¹`. **Re-read CONTEXT.md Section 5-6 if any of the above is unclear**, before continuing.

---

## Layer Selection (Provisional)

The user originally planned to target `layer 19, neuron 770` of GPT2-medium per Lee et al. 2024. **That probe-based identification did not transfer to GPT-Neo-125M directly, but the *position* of toxic mechanism in transformer depth is architecture-invariant**: Lee et al.'s top-7 toxic value vectors concentrate in layers 18-20 out of 24 (75%-87% of depth). By analogous proportion in GPT-Neo-125M's 12 layers, toxic vectors should concentrate around layers 8-10.

**For Phase 2 development, USE THIS PROVISIONAL TARGET**:
- Primary: `transformer.h.9.mlp.c_proj` (layer 9 of 12, ≈75% depth — matches Lee's median layer 19/24)
- Shape: (3072, 768) — d_mlp × d_model
- Rationale: corresponds proportionally to where Lee et al. found toxic vectors in GPT2-medium

**Secondary candidate** (for later layer-selection comparison): `transformer.h.10.mlp.c_proj` (layer 10, ≈83% depth).

When the user finalizes layer selection (post-PPO, based on either weight-delta analysis or IF discriminability metrics), swapping is a one-line config change — your code MUST support this via `config.layer_name` parameter. Do NOT hardcode `"h.9.mlp.c_proj"` anywhere except `config.py`.

---

## Hard Constraints (Phase 2 specific — must hold, no exceptions)

1. **Prompt masking**: Loss / gradient computations must mask out prompt tokens (the per-token contribution at prompt positions is zero, response positions contribute). This applies to:
   - Stage 1 (Kronecker factor accumulation)
   - Stage 3 (per-sample gradient for IF score)

2. **Per-sample gradient in Stage 3**: Use `batch_size=1` for the per-sample gradient computation. Do NOT use batched `autograd.grad` — it returns mean gradients, not per-sample. Verify this by comparing batched vs loop output on at least 3 samples; numbers must match to fp32 precision.

3. **Reduction**: All cross-token reductions use `mean per-token` (not `sum`). This applies to both Stage 1 accumulation and Stage 3 IF computation. Mismatched reductions produce length-biased IF scores.

4. **Stage 1 pseudo labels, Stage 3 recorded labels**:
   - Stage 1 (estimating Fisher) samples pseudo-labels from the model's own distribution `y ~ π_θ*(·|x)` to estimate the *true* Fisher.
   - Stage 3 (computing IF for a specific rollout) uses the *recorded* label `y_m` from disk — that rollout's actual response.

5. **EK-FAC validation gates** (revised based on Stage 2 findings):
   - Implementation correctness: 4 Stage 2 gates at machine precision 
     (Λ-vs-diag < 1e-10, synthetic-independence reconstruction < 1e-8,
     Q orthonormality < 1e-10, A/S reconstruction < 1e-6)
   - At-scale validation: real-data self-influence sanity (5/5 
     sign-consistent, top-5 magnitude)
   - LOO retraining acknowledged as infeasible per Bae et al. 2022
   - Original 1e-3 Frobenius gate against brute-force F⁻¹ is 
     structurally unattainable on transformer EK-FAC (~45% off-diagonal 
     Kronecker mass); replaced by above gates

6. **Real-data mid-pipeline gate**: After toy passes AND before declaring Phase 2 done, your pipeline must produce sensible numbers on existing `data/rollouts/step_0050.pt` (or earliest available step file). "Sensible" defined per stage below.

7. **No imports from `references/`**: read-only reference, re-implement.

8. **Stage gates**: each stage must pass its acceptance criteria before the next stage starts. Do not write Stage 2 code while Stage 1 is failing.

---

## Pipeline Structure (Stages)

### Stage 0: Data loaders & toy model
- `src/ekfac/data.py`: Load rollouts from `data/rollouts/`, batch them, apply prompt mask
  - Input: directory of step_NNNN.pt files
  - Output: iterable of `(prompt_ids, response_ids, response_mask)` tuples
- `src/ekfac/toy.py`: A tiny transformer for validation
  - Architecture: d_model=8, d_mlp=16, n_layer=2, vocab_size=20, T_max=4
  - Same MLP structure as GPT-Neo's `c_proj` (Linear from d_mlp to d_model, no bias optional)

**Acceptance**:
- `tests/test_data.py` loads ≥ 1 real rollout file, returns correct shapes
- `tests/test_toy.py` instantiates toy model, forwards a random input, returns logits of expected shape

### Stage 1: Kronecker Factor Accumulation
- `src/ekfac/hooks.py`: forward + backward hook utilities
- `src/ekfac/factors.py`: accumulate A and S over data
  - `A = E[a aᵀ]` where `a` is the input to c_proj (shape d_mlp × d_mlp)
  - `S = E[g gᵀ]` where `g` is the output gradient of c_proj (shape d_model × d_model)
- Run forward+backward on **pseudo-labeled** data (sample `y ~ π_θ*(·|x)`, then compute `∇ log π_θ*(y|x)`)

**Acceptance — toy model**:
- A, S are symmetric to fp64 (max asymmetry < 1e-10)
- A, S are positive semi-definite (min eigenvalue ≥ -1e-8, allow tiny negative from float roundoff)
- Diagonal of A and S is strictly positive
- Print these three values and the eigenvalue ranges of A and S

**Acceptance — real data on `data/rollouts/step_0050.pt`** (or earliest available step file):
- Accumulate over the 32 rollouts in that file
- A, S still symmetric and PSD
- Eigenvalue ranges reasonable: largest eigenvalue of A is `O(1)` to `O(100)`, largest of S is `O(0.001)` to `O(0.1)` (rough — surface anything wildly outside this; the surprise itself is informative)
- Stage 1 on 32 rollouts takes < 5 minutes on GPU

### Stage 2: Eigendecomposition + Lambda Diagonal
- `src/ekfac/eigen.py`:
  - Decompose `A = Q_a Λ_a Q_aᵀ`, `S = Q_s Λ_s Q_sᵀ` (use `torch.linalg.eigh`)
  - Estimate Lambda diagonal: `Λ_ij = E[(Q_aᵀ a)_i² (Q_sᵀ g)_j²]`
    - This is the per-eigenvector pair variance; requires a second pass over data with the eigenvectors fixed
- Output: `Q_a.pt`, `Q_s.pt`, `Lambda.pt`

**Acceptance — toy model basic**:
- Λ_a and Λ_s eigenvalues are sorted descending
- Q_a Q_aᵀ ≈ I and Q_s Q_sᵀ ≈ I to fp32
- Lambda matrix has shape (d_mlp, d_model) = (16, 8) for toy, all entries > 0
- The reconstruction `Q_a · diag(Λ_a) · Q_aᵀ ≈ A` to relative error 1e-6

**Acceptance — toy model: EK-FAC vs True Fisher**:
- Compute true Fisher `F_true ∈ R^(d_model*d_mlp × d_model*d_mlp) = R^(128 × 128)` for toy via direct enumeration: forward each pseudo-sample, get `∇log p`, accumulate `∇log p · ∇log pᵀ` outer products
- Compute `F_true_inv = np.linalg.inv(F_true + damping * I)` with damping = 0.01
- Compute EK-FAC's `F_inv_ekfac` (which is the inverse via the eigendecomp scaled by `1/(Λ + damping)`)
- Compare: `||F_inv_ekfac - F_true_inv||_F / ||F_true_inv||_F < 1e-3`
- **If this fails, stop and debug. Do NOT proceed to Stage 3 on real model.**

### Stage 3: Per-Sample Gradient Projection + IF Score
- `src/ekfac/influence.py`:
  - For each rollout `z_m`:
    - Compute `∇_φ log π_θ*(y_m | x_m)` (batch_size=1, true per-sample gradient)
    - Project: `G_m = Q_sᵀ · gradient · Q_a` (gradient is shape (d_model, d_mlp), G_m is same shape)
    - Scale: `G_m_scaled = G_m / (Λ + damping * I)` element-wise (Λ is shape (d_mlp, d_model))
  - For eval objective `f`:
    - Compute `∇_φ f` similarly
    - Project: `G_f = Q_sᵀ · ∇_φ f · Q_a`
    - **Note: do NOT scale G_f by Lambda — only G_m gets scaled. Verify this matches CONTEXT.md Section 5-6's formula.**
  - IF score: `I(z_m, f) = -⟨G_f, G_m_scaled⟩` (Frobenius inner product, scalar)

**Acceptance — toy model**:
- Compute IF via EK-FAC (just-built pipeline) and via brute-force true Fisher inverse
- Per-rollout IF values match within relative error 5% (looser than Stage 2 because Stage 3 compounds Stage 2 error with per-sample gradient approximation)
- Sign of IF matches between methods for all 10+ test rollouts (rank correlation Kendall's tau > 0.95)

**Acceptance — real data on `data/rollouts/step_0050.pt`**:
- Compute IF for all 32 rollouts in that file, against a default eval objective (use 1 rollout from the same file as the eval target — self-influence sanity check)
- Print IF score distribution: min, max, median, std
- IF scores should span at least 2 orders of magnitude (otherwise pipeline is broken — all rollouts can't be equally influential)
- Self-influence of the eval rollout (one rollout used as both training and eval) must be positive and among the top-5 IF scores

---

## File Structure to Create

```
src/ekfac/
├── __init__.py
├── config.py       # layer_name, damping, dtype, paths — all parameterized
├── data.py         # rollout loader, prompt/response mask
├── toy.py          # toy transformer for validation
├── hooks.py        # forward/backward hook registration
├── factors.py      # Stage 1: A, S accumulation (pseudo-labeled)
├── eigen.py        # Stage 2: eigendecomp + Lambda diagonal
└── influence.py    # Stage 3: per-sample gradient + IF score

tests/
├── test_data.py              # rollout loading, mask
├── test_toy.py               # toy model forward
├── test_factors_toy.py       # Stage 1 acceptance on toy
├── test_factors_real.py      # Stage 1 acceptance on real rollouts
├── test_eigen_toy.py         # Stage 2 acceptance on toy + true Fisher comparison
├── test_influence_toy.py     # Stage 3 acceptance on toy + true IF comparison
└── test_influence_real.py    # Stage 3 acceptance on real rollouts

src/run_ekfac.py              # orchestrating script (does not need to be perfect)
```

---

## Engineering Constraints

- **Dtype**: Use float64 for A, S, Lambda accumulation. Use float32 for per-sample gradients during projection and IF computation.
- **GPU memory**: GPT-Neo-125M fp32 is ~500 MB; A (3072×3072 fp64) is 70 MB; S (768×768 fp64) is 4.5 MB. Trivially fits in 48 GB. Don't optimize for memory.
- **GPU sharing**: PPO is using ~14 GB on the GPU. You have 34 GB free. Don't allocate huge buffers without checking.
- **Speed**: Stage 1 over 1000 rollouts in a real run should take 10-30 minutes. Stage 3 per-rollout is ~0.5-2 seconds. Don't optimize unless these are exceeded.
- **Damping `λ`**: Start with `λ = 0.01`. Note: damping is applied as `1/(Λ + λ * I)` in eigenspace, equivalent to `(F + λ*I)⁻¹` for the original Fisher. Don't add damping in two places — Stage 2 only.
- **Determinism**: All random sampling (pseudo-labels, batch shuffling) uses `torch.manual_seed(42)`. Pseudo-label sampling temperature = 1.0 (sample from true `π_θ*`, not greedy).
- **Logging**: Save to `data/ekfac/` directory. Subdirectories: `factors/`, `eigen/`, `influence/`. Use `<stage>_<layer_name>_<timestamp>.pt` naming so multi-layer experiments don't collide.

---

## Reporting Protocol

Phase 2 follows Phase 0's "stop and report at stage boundaries" pattern.

After each Stage acceptance test, **STOP coding and surface a report** with:
1. What was implemented (1-2 sentences)
2. Acceptance metric values (the actual numbers, not just "pass")
3. Any surprises or anomalies
4. Proposed next stage

Do NOT proceed past a stage's acceptance gate without the user's explicit "approved, proceed to Stage N+1".

If an acceptance gate fails, **DO NOT** silently lower the bar. Surface the failure with:
- What you measured vs the threshold
- What you've already tried
- Three hypotheses for the cause
- What evidence would discriminate between them

This rule exists because Phase 0 had multiple instances of agents silently relaxing thresholds. The user explicitly does not want this pattern.

---

## What NOT to do

- Do not touch `data/rollouts/`, `data/ppo_checkpoints/`, `data/ppo_train_log.jsonl` — Phase 0 is writing to these files.
- Do not modify `src/train_ppo.py` — Phase 0 owns it.
- Do not modify `CONTEXT.md` — user owns the theory doc. If you find a discrepancy, surface it.
- Do not import from `references/` — read-only.
- Do not optimize for speed before correctness.
- Do not skip the toy validation gate even if it feels slow — without it, real-model bugs are undetectable downstream.
- Do not silently change acceptance thresholds.
- Do not hardcode the layer name — must be in `config.layer_name`.

---

## First-Day Action Items

1. **Read** `CONTEXT.md` in full (it's in the project root).
2. **Read** `CLAUDE.md` in full.
3. **Skim** `references/MDA-EK-FAC/` — pattern-only, no copy.
4. **Inventory**: List the rollout files currently in `data/rollouts/` (PPO is running so files are accumulating). Open one (e.g., `data/rollouts/step_0050.pt` if it exists, else the latest available) and print the schema to confirm it matches this doc.
5. **Setup**: Create `src/ekfac/` skeleton with empty modules. Write `config.py` with the provisional layer (`transformer.h.9.mlp.c_proj`) and damping (0.01).
6. **Toy model**: Implement the toy transformer (`src/ekfac/toy.py`) + the brute-force true Fisher computation script (in `tests/test_eigen_toy.py`). This is the hardest piece because it requires you to derive the true Fisher correctly and verify it matches `autograd.grad` outputs.

**STOP after step 6 and surface a report** with:
- Confirmed rollout schema (paste output of one `torch.load(...).keys()` per rollout dict)
- Toy model architecture summary
- True Fisher computation script (paste the code)
- One concrete numerical sanity check (e.g., F_true is symmetric, positive semi-definite, condition number)

The user will then approve Stage 1 implementation.

---

Read this doc, then say **"Phase 2 kickoff received, starting required reading."** Do not write any code yet. After completing required reading (items 1-4 above), say **"Required reading complete, ready for Action Items 5-6."** Then proceed.
