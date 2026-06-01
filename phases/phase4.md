# Phase 4 — RL-KL 影响函数的理论误差预算与稳健性（完善版）
# Theoretical Error-Budget & Robustness of the RL-KL Influence Function

> 给云端 session 的说明：这是一份自包含的实验 spec，目标机器是 `/root/rlhf-influence`（GPT-Neo-125M 去毒 PPO 实验，4090 48G）。先读 repo 根目录的 `CLAUDE.md` 与 `CONTEXT.md` 把约定（θ\*、φ、β、k1、prompt mask、pseudo-label、batch=1 等）对齐，再实现下面的实验。所有实验复用已有的 Phase 1 / Phase 2 代码（`src/ekfac/`, `src/phase1/`）。计算资源不设限。
>
> **本版相对初稿的关键修订**（每处在正文用【修订】标注）：
> 1. E2 的 $\|G\|^2$ 必须用 split-half 无偏估计，否则 2.36M 维下天真 $\|\hat G\|^2$ 几乎全是噪声、会错误锤死非平稳性。
> 2. E1 去掉概念错误的"$\tilde R$ across-run 噪声地板"（固定 $(x,y,\theta^*)$ 下 $\tilde R$ 是确定量），改为 ICC 估计误差带 + 干净的 reward/KL 方差分解。
> 3. E3 厘清一阶 Neumann 的可信域：一阶结论仅在 $\|\tilde\Delta p\|/\|g_f\|$ 小时可信；大则必须升级全阶。$\|\tilde\Delta p\|^2$ 同样需 split-half 去偏。
> 4. E3 全阶 solver 用 MINRES（$F-\tilde\Delta$ 对称不定），不是 CG/GMRES。
> 5. 多处填全采样数 / seed / 公式方向 / 输出 schema / 收敛检查。

---

## 0. 目标：锤死 or 转机

毕业论文 IF 公式依赖强假设 **A2：$\pi_{\theta^*}=\pi^*$**。Phase 1 测得 ICC=0.892，**A2 在 $y$ 层级不成立**。

本实验要回答的不是"再确认一次 A2 失败"，而是**定量问题：A2 失效把 IF 公式损坏了多少？** 中心任务（E3）测量修正项 $\Delta$ 相对 $\beta F$ 的大小，以及加/不加 $\Delta$ 修正后 per-rollout IF 排序是否改变。

- **锤死**：$\Delta/\beta$ 相对 $F$ 不可忽略，且加修正后排序显著变（Spearman 明显 < 1）→ 论文那条干净公式数值上不成立。
- **转机**：$\Delta/\beta$ 相对 $F$ 小、排序几乎不变（Spearman ≈ 1）→ 尽管 ICC=0.892，丢掉 $\Delta$ 是一阶精确近似，IF 排序成立，且能给出有误差界的局部归因声明。

> 给云端 session：**不要带预期。诚实测 $\Delta$ 的大小和它对排序的影响，让数字说话。所有"距离最优""误差小"之类是对数值的解读，必须与实测数字分开写。**

---

## 1. 理论回顾

记号沿用论文 / `CONTEXT.md`：prompt $x$，rollout $y=(y_1..y_T)$，$\pi_\theta(y|x)$，$\pi_\text{ref}$，奖励 $r(x,y)$（RoBERTa 无毒 logit），KL 系数 $\beta$。φ = layer 9 MLP $W_2$（`transformer.h.9.mlp.c_proj`，shape (3072, 768)，约 2.36M 参数）。score function $s(x,y):=\nabla_\phi\log\pi_\theta(y|x)$。

$$J(\theta)=\mathbb{E}_{x,\,y\sim\pi_\theta}[\tilde R],\quad \tilde R := r - \beta\log\tfrac{\pi_\theta}{\pi_\text{ref}},\qquad G(\theta):=\nabla_\phi J = \mathbb{E}_{x,\,y\sim\pi_\theta}[s\cdot(\tilde R-\beta)]$$

**A2 成立时**：Boltzmann 结构使 $\tilde R(x,y,\theta^*)=\beta\log Z(x)$ 对 $y$ 为常数，项 A（Fisher 形）与项 B（log-policy Hessian 形）精确相消，$\partial G/\partial\theta|_{\theta^*}=-\beta F(\theta^*)$，IFT 给出论文公式 $\mathcal I_f(z_m)=-g_f^\top F^{-1}s_m$。

**A2 不成立时**：定义有效奖励 advantage（同 prompt 内中心化 $\tilde R$）
$$A(x,y):=\tilde R(x,y,\theta^*)-\mathbb{E}_{y'\sim\pi_{\theta^*}(\cdot|x)}[\tilde R(x,y',\theta^*)]$$
项 A、B 不再精确相消，留下
$$\boxed{\;\frac{\partial G}{\partial\theta}\Big|_{\theta^*}=\Delta-\beta F,\qquad \Delta:=\mathbb{E}_{x,\,y\sim\pi_{\theta^*}}\big[A(x,y)\cdot(s\,s^\top+\nabla^2_\phi\log\pi_{\theta^*})\big]\;}$$
只有随 $y$ 变化的 $A$ 残留进 $\Delta$；常数部分 $\mathbb{E}[\tilde R]$ 仍相消。ICC=0.892 量的是 $\mathrm{Var}_y[\tilde R]\propto\mathbb{E}[A^2]$——$A$ 不为零，但 $\Delta$ 是否大是另一回事。

修正版 IF：
$$\mathcal I_f^{\text{corr}}(z_m)=-g_f^\top(\beta F-\Delta)^{-1}\beta s_m=-g_f^\top(F-\tilde\Delta)^{-1}s_m,\qquad \tilde\Delta:=\Delta/\beta$$
论文公式 = 修正公式在 $\tilde\Delta=0$ 的特例。关键无量纲量是 $\tilde\Delta$ 相对 $F$ 的大小。

**【修订 · $\tilde\Delta$ 的关键性质，实现 E3 前必读】**
- $\tilde\Delta$ **对称**（$ss^\top$ 与 $\nabla^2\log\pi$ 都对称）但**可能不定**（$A$ 可正可负）。故全阶解 $(F-\tilde\Delta)p_\text{corr}=g_f$ 用 **MINRES**（对称不定），不是 CG（需正定）也不是 GMRES（非对称才需要）。
- 一阶 Neumann 的真实收敛参数是谱半径 $\rho(F^{-1}\tilde\Delta)$，**不是**之前算的标量 $|\delta|/\beta\approx6.5$。后者是粗糙上界；因为 $A$ 逐 prompt 零均值、加权曲率大量抵消，加上 $F^{-1}$ 预条件，实际算子效果可能远小于 6.5。E3 直接测 $\|\tilde\Delta p\|/\|g_f\|$ 就是量真实算子效果，是比标量界更可信的判据。

**两个独立误差源（务必都报告）**：
1. **平稳性误差**：θ\* = step_0650 是 Pareto checkpoint，不是 $J$ 的平稳点，$G(\theta^*)\neq0$。IFT 本要求 $G=0$。→ E2 量化。
2. **曲率误差（中心）**：即便假装 $G=0$，$\partial G/\partial\theta=\Delta-\beta F\neq-\beta F$。→ E3 量化。

---

## 2. 环境与约定（实现前核对）

- **repo** `/root/rlhf-influence`；先读 `CLAUDE.md`+`CONTEXT.md`。
- **θ\*** `data/ppo_checkpoints/step_0650/`（125,198,592 参数）。
- **φ** layer 9 MLP `W_2` = `transformer.h.9.mlp.c_proj`（3072→768，约 2.36M）。
- **β** = **0.2365**（step 650 自适应 KL 系数，与 Phase 1 一致）。
- **KL 估计器** 默认 **k1**（与训练+Phase 1 一致）；E5 对比 k3。
- **rollouts** `data/rollouts/step_NNNN.pt`，生产区间 step 0..650 = **20,832 条**。
- **固定评估集** `data/eval_prompts_400.json`（400 条）。
- **Phase 1 数据** `data/phase1/`（100 prompts × 32 rollouts 的 $\tilde R$ 方差诊断、ICC=0.892）。
- **Phase 2 / EK-FAC** `data/ekfac/`（184G：A/S 因子、特征分解、$\Lambda$、IHVP、production influence、4 个评估目标 $f_\text{seq},f_\text{toxic}^{C1/C2/C3}$ 的 per-rollout 分数与缓存的 $s_m$、$g_f$、$p=F^{-1}g_f$）。
- **硬约束**（沿用 `CLAUDE.md`）：prompt mask `ignore_index=-100`；per-sample 梯度 batch=1；Fisher 用 pseudo-label（从 $\pi_\theta$ 采样）；reduction=mean；fp64 累积 A/S/Λ。
- **可复用算子**：matrix-free $Fv$ 和 IHVP（$F^{-1}v$，EK-FAC + 两级 damping `max(Λ+0.1·Λ̄, 1e-5)`）。

**【修订 · 全局统计纪律，所有实验适用】**
> 凡是要报告一个高维向量的范数平方（$\|G\|^2$、$\|\tilde\Delta p\|^2$）或二次型（$G^\top F^{-1}G$），**禁止**用单批样本的 $\|\hat v\|^2$——它含 $+\frac1n\mathrm{tr}(\mathrm{Cov})$ 的自积偏差，在 2.36M 维下几乎全是噪声。**一律用 split-half 无偏估计**：把样本随机分两不相交半 $A,B$，独立算 $\hat v_A,\hat v_B$，用 $\langle\hat v_A,\hat v_B\rangle$ 估 $\|v\|^2$（二次型用 $\hat v_A^\top M\hat v_B$）。重复 ≥20 次随机 split 取均值 + 给 bootstrap CI。

**统一随机种子**：on-policy 采样主 seed = 42；split / bootstrap 用 seed 1000+i。所有实验记录实际 seed 到报告。

---

## 3. 实验

每个实验给：原理 / 计算 / 测什么 / 判据 / 优先级 / 成本。

---

### E1（P0，前置，便宜）— 有效奖励 advantage $A(x,y)$ 的分布 + ICC 的方差分解

**原理**：$\Delta$ 由 $A(x,y)$ 加权曲率构成，先刻画 $A$ 分布。并回答：ICC=0.892 里多少是"真实策略次优"、多少是 reward / KL 结构性方差。

**计算**：
1. 取 Phase 1 的 100 prompts × 32 rollouts（训练分布内）。对每条算
   $$\tilde R(x,y,\theta^*)=r(x,y)-\beta\sum_t[\log\pi_{\theta^*}(y_t|\cdot)-\log\pi_\text{ref}(y_t|\cdot)]\quad(\text{k1，复用 Phase 1})$$
2. $A(x,y)=\tilde R-\overline{\tilde R}_x$（同 prompt 内中心化）。报告 $A$ 分布：均值（应≈0）、std、峰度、偏度、非对称长尾的量化（Phase 1 已观察到，给 Q-Q 图与 top/bottom-5% 占比）。
3. **【修订 · 干净的方差分解，E1 主结果】** 把 within-prompt 方差按 reward 与 KL 两个来源精确分解：
   $$\mathrm{Var}_y[\tilde R]=\mathrm{Var}_y[r]+\beta^2\mathrm{Var}_y[\mathrm{KL}]-2\beta\,\mathrm{Cov}_y[r,\mathrm{KL}]$$
   其中 $\mathrm{KL}=\sum_t[\log\pi_{\theta^*}-\log\pi_\text{ref}]$（per-rollout 求和）。逐 prompt 算三项，报告三项在总 within 方差中的平均占比。这给出"方差到底来自 reward 噪声还是 KL 偏离"的 principled 拆分。
4. **长度 lens（次要）**：保留 Phase 1 的 $\mathrm{corr}(\text{length},\tilde R)$ 与 $\rho^2\approx0.11$，作为对 step 3 分解的补充解释（长度同时进 reward 与 KL，主要体现在 Cov 项）。
5. **【修订 · ICC 估计误差带，替代原"噪声地板"】** 原"固定 response 重采"概念上错误（固定 $(x,y,\theta^*)$ 下 $\tilde R$ 是确定量，无随机性）。改为：选 ~20 条 prompt，用不同 seed 各**重抽 32 条新 response**，每轮重算 within-prompt 方差与整体 ICC，报告 ICC=0.892 的 run-to-run **估计标准差 / CI**。这量的是"ICC 统计量本身的抽样误差"，不是 $\tilde R$ 的噪声。
   > 诚实说明：within-prompt 方差中**没有**一个独立于 $A$ 的"不可消噪声地板"——$\mathrm{Var}_y[\tilde R]$ 按定义就是 A2 失效信号。RoBERTa 对近义改写的脆弱性确实可能贡献一部分（同义不同分），但它属于 reward 定义的一部分、是 IF 合法归因的对象，不应被当作噪声抹掉。这点要写清，不要用"RM 噪声"给 A2 失效打折。

**测什么**：$A$ 分布统计；reward/KL/Cov 三项占比；length $\rho^2$；ICC 估计 CI。

**判据**：若 within 方差主要由 $\mathrm{Var}_y[r]$（reward 结构）+ 长度主导、KL 偏离项相对小 → A2 失效里有相当部分是 reward 面本身的形状而非 policy 在 KL 度量上的偏离 → 偏转机的一个线索（但不是决定性，决定性看 E3）。反之偏锤死。

**输出**：`data/phase4/e1_advantage_stats.json`（$A$ 矩、分解占比、ICC CI）、`reports/phase4/e1_*.png`（$A$ 分布、Q-Q、三项占比堆叠图）。

**成本**：低（复用 Phase 1 + 一轮重采样）。

---

### E2（P0，便宜）— $\|G(\theta^*)\|$：θ\* 的一阶非平稳度

**原理**：IFT 要求 $G(\theta^*)=0$。θ\* 是 Pareto checkpoint，不是平稳点。量它离平稳多远。

**计算**：
1. **on-policy** 采样（注意是 $y\sim\pi_{\theta^*}$，不是用已存训练 rollout）：从训练 prompt 池采 $N=256$ prompt，每个用 θ\* 采 $K=8$ 条 rollout（共 2048 样本）。对每条算 $s=\nabla_\phi\log\pi_{\theta^*}$ 和标量 $(\tilde R-\beta)$。
2. **【修订 · split-half 无偏估计 $\|G\|^2$】** 禁止直接算 $\|\frac1n\sum s_i(\tilde R_i-\beta)\|^2$（含 $+\frac1n\mathrm{tr}(\mathrm{Cov})$ 偏差，2.36M 维下几乎全噪声）。改为：随机把 2048 样本分两半，独立算 $\hat G_A,\hat G_B$，$\|G\|^2\approx\langle\hat G_A,\hat G_B\rangle$。重复 ≥20 次 split 取均值 + bootstrap CI。判断 $\|G\|$ 是否显著非零。
3. **自然梯度步长** $\sqrt{G^\top F^{-1}G}$：同样用 split-half 去偏，$G^\top F^{-1}G\approx\hat G_A^\top F^{-1}\hat G_B$（$F^{-1}\hat G_B$ 用已有 IHVP）。它近似"θ\* 沿牛顿方向走一步移动多少 KL"。
4. 对比训练时单步典型 KL（wandb 日志 per-step KL / target KL=6.0）。

**测什么**：$\|G(\theta^*)\|$（split-half）及 CI；$\sqrt{G^\top F^{-1}G}$ vs 训练单步 KL。

**判据**：自然梯度步长 ≪ 训练单步 KL → θ\* 近平稳，平稳性误差小（转机）；≳ 训练单步 KL → 实质非平稳，IF 反事实解释受限（锤死，但仍可走 PBRF 局部排序解释）。

**输出**：`data/phase4/e2_stationarity.json`（split-half $\|G\|^2$ 序列、CI、自然梯度步长、对比 KL）。

**成本**：低-中（2048 次 backward + 几十次 IHVP）。

---

### E3（P0，中心实验）— $\Delta$ 误差预算：加/不加修正，IF 排序是否改变

**原理**：直接测"丢掉 $\Delta$"对 per-rollout IF 排序的影响。一阶 Neumann 给主导项，必要时全阶 MINRES 确认。

**关键代数（照此实现）**：$p:=F^{-1}g_f$（Phase 2 已为每个 $f$ 算过，复用；论文分数 $\mathcal I_f(z_m)=-p^\top s_m$）。一阶展开 $(F-\tilde\Delta)^{-1}\approx F^{-1}+F^{-1}\tilde\Delta F^{-1}$：
$$\mathcal I_f^{\text{corr}}(z_m)\approx\underbrace{-p^\top s_m}_{\text{论文}}-\underbrace{q^\top s_m}_{\text{一阶修正}},\qquad q:=F^{-1}\,\tilde\Delta\,p$$
推导：修正项 $=-g_f^\top F^{-1}\tilde\Delta F^{-1}s_m=-(F^{-1}\tilde\Delta p)^\top s_m=-q^\top s_m$（用 $g_f^\top F^{-1}=p^\top$、$\tilde\Delta$ 对称）。**只需一次 $\tilde\Delta p$（一次 $\Delta$-向量积）+ 一次 IHVP 得 $q$，再对 2 万条 rollout 做点积 $-q^\top s_m$——与主打分一样便宜。**

**$\tilde\Delta$-向量积的 matrix-free 实现**（对任意 $v$）：
$$\tilde\Delta v=\frac1\beta\,\mathbb{E}_{x,\,y\sim\pi_{\theta^*}}\Big[A(x,y)\cdot\big(s\,(s^\top v)+\mathrm{HVP}_{\log\pi}(v)\big)\Big]$$
- $A(x,y)$：标量，需同 prompt 多 rollout 估 $\overline{\tilde R}_x$，故 on-policy 采 $N$ prompts × $K$ rollouts。
- $s(s^\top v)$：$s^\top v$ 标量内积，便宜。
- $\mathrm{HVP}_{\log\pi}(v)=\nabla^2_\phi\log\pi_{\theta^*}(y|x)\,v$：double-backward（`torch.autograd.grad(grad(logπ,φ,create_graph=True)·v, φ)`），每样本约 3× backward。
- **【修订 · pseudo-label 与 A 同一次抽样】** $F$ 的 pseudo-label 与 $A(x,y)$ 用的 response 必须是**同一次 on-policy 抽样**的同一个 $y$——$y\sim\pi_{\theta^*}$ 既作 Fisher 的 pseudo-label、又用其真实 RoBERTa 奖励算 $A$。**不要分两次采**，否则 $F$ 与 $\Delta$ 的度量不一致。

**计算步骤**：
1. **on-policy 采样池**：$N=200$ prompts（训练分布内）+ 另取 100（训练分布外做稳健性）× $K=16$ rollouts from θ\*。算每条 $s$、$\tilde R$、$A$。**【修订】** 记录这批样本，E1/E2/E3 尽量共用同一池以保证一致。
2. 对每个 $f\in\{f_\text{seq},f_\text{toxic}^{C1},f_\text{toxic}^{C2},f_\text{toxic}^{C3}\}$：取已有 $p=F^{-1}g_f$。
3. 算 $\tilde\Delta p$（matrix-free，在池上平均）。**【修订 · 收敛 + 去偏检查】** 把采样池分两半算 $\tilde\Delta_A p,\tilde\Delta_B p$：(a) $\|\tilde\Delta p\|^2\approx\langle\tilde\Delta_A p,\tilde\Delta_B p\rangle$（split-half 去偏）；(b) 报告两半的 $\tilde\Delta p$ 余弦相似度作收敛指标（低则需加大 $N,K$）。
4. **标量诊断**：$\|\tilde\Delta p\|/\|g_f\|$（注意 $Fp=g_f$ 故 $\|g_f\|=\|Fp\|$）——$\Delta$ 在"真正用到的方向 $p$"上相对 $F$ 的扰动幅度。另用 Hutchinson（几个随机 $v$）报告 $\|\tilde\Delta\|/\|F\|$ 的全局标度。
5. 算 $q=F^{-1}(\tilde\Delta p)$（一次 IHVP），对 20,832 条 rollout 算 $\mathcal I_f^\text{corr}(m)=-p^\top s_m-q^\top s_m$（$s_m$ 复用 Phase 2 缓存）。
6. **对比**：Spearman$(\mathcal I_f,\mathcal I_f^\text{corr})$、Pearson、top-10/50/100 Jaccard、修正项相对量 $\mathrm{median}_m|q^\top s_m|/|p^\top s_m|$。4 个 $f$ 分别做。
7. **【修订 · 全阶仅在边界触发，且用 MINRES】** 仅当某 $f$ 上一阶 Spearman 落入 0.9–0.98 边界 **或** $\|\tilde\Delta p\|/\|g_f\|\gtrsim0.3$ 时，上全阶：matrix-free $(F-\tilde\Delta)v$ 跑 **MINRES**（对称不定）解 $(F-\tilde\Delta)p_\text{corr}=g_f$，得全阶修正排序确认一阶结论。用与主管线一致的 damped $F$。记录 MINRES 残差与迭代数。

**测什么**：(a) $\|\tilde\Delta p\|/\|g_f\|$（split-half）与 $\|\tilde\Delta\|/\|F\|$；(b) 4 目标 Spearman/Jaccard（论文 vs 修正）；(c) per-rollout 修正项相对幅度分布。

**判据（中心）**：
- **转机**：4 目标全部 Spearman$>0.95$ 且 top-50 Jaccard 高 且 $\|\tilde\Delta p\|/\|g_f\|\lesssim0.1$ → 丢 $\Delta$ 是一阶精确近似，IF 排序数值上成立（尽管 ICC=0.892）。把修正项相对幅度作"误差棒"，给排序一个被验证的局部精度声明。
- **锤死**：任一关键目标 Spearman$<0.9$ 或 $\|\tilde\Delta p\|/\|g_f\|\gtrsim0.3$ → 丢 $\Delta$ 不合法。**【修订】** 此时**一阶 Spearman 本身不可信**（高阶项主导），不能拿一阶结论下定论，必须以全阶 MINRES 结果为准；并诚实报告是哪个 $f$、哪些 rollout（很可能是 E1 长尾 $A$ 大的 prompt）最受影响。
- 中间地带：报临界值，分目标标注稳/不稳。

**输出**：`data/phase4/e3_delta_budget.json` + `reports/phase4/e3_budget_table.md`（4×{Spearman, Jaccard@10/50/100, $\|\tilde\Delta p\|/\|g_f\|$, 修正项相对幅度分布} 表 + split-half 收敛指标 + 任何 MINRES 触发记录）。

**成本**：一阶便宜（1 个 $\Delta$-VP + 1 IHVP + 2 万点积，每目标）；HVP double-backward 在 $N\times K$ 样本上 4090 数小时内。全阶 MINRES 仅边界目标。

---

### E4（P1，中）— 收敛交叉验证：A2-无关归因 vs EK-FAC IF

**原理**：若一个**完全不依赖 A2** 的归因方法在同一批 rollout 上给出和 EK-FAC IF 一致的排序，则 A2 失效在实践中没损坏排序——独立佐证 E3。

**计算**（同 20,832 条 rollout、同 4 个 $f$，与论文 IF 比 Spearman/Jaccard）：
1. **Gradient similarity（必做，近零成本）**：$\mathcal I_\text{grad}(m)=-g_f^\top s_m$（不带 $F^{-1}$）。Phase 2 已有 $g_f,s_m$。差异纯来自 $F^{-1}$——若两者 Spearman 高，说明 Fisher 预条件不决定性，$\Delta$（对 $F$ 的扰动）更不可能翻盘。
2. **Bayesian-IF / SGLD（可选，较贵，仅 E3 边界时做）**：localized Gibbs 后验 + SGLD 采样 + 损失协方差 $-\mathrm{Cov}_\gamma(\ell_m,f)$，无 Hessian/Fisher、不假设最优。在 φ 子空间做。

**判据**：高一致 → 排序稳健（转机加强）；低一致 → 需进一步查哪种对。

**输出**：`data/phase4/e4_crossval.json`。

**成本**：grad-sim 近零；Bayesian-IF 中等。

---

### E5（P1，便宜）— k1 vs k3 KL 估计器对 $\tilde R$ / ICC / IF 排序的影响

**原理**：论文用 k1（有符号、被压制 token 上变成奖励）。k3 无偏+低方差+恒正、是产线标准。测换 k3 后 ICC 与 IF 排序变不变，把"k1 隐患"量化。

**【修订 · k3 公式与方向写死】** 估计 $\mathrm{KL}(\pi_\theta\|\pi_\text{ref})$、样本来自 $\pi_\theta$，令 per-token ratio $\rho_t=\pi_\text{ref}(y_t|\cdot)/\pi_{\theta^*}(y_t|\cdot)$（即 $\log\rho_t=\log\pi_\text{ref}-\log\pi_{\theta^*}$）：
$$\widehat{\mathrm{KL}}_{k3}=\sum_t\big[(\rho_t-1)-\log\rho_t\big]\ge0$$
（对照 k1：$\widehat{\mathrm{KL}}_{k1}=\sum_t(-\log\rho_t)=\sum_t[\log\pi_{\theta^*}-\log\pi_\text{ref}]$。）

**计算**：
1. 用 k3 重算 E1 的 $\tilde R,A,\mathrm{ICC}$，对比 k1 的 ICC=0.892。
2. （若 E3 已搭好）用 k3 的 $\tilde R$ 重算 $A,\Delta$，看 $\|\tilde\Delta p\|/\|g_f\|$ 变不变。
3. 说明：KL 估计器只进 $\tilde R$（从而 $A,\Delta$）；$F$（纯 score 外积）不依赖它。IF 公式形式不变，变的是"A2 失效程度"。

**判据**：k3 显著降低 within 方差 → 论文"A2 失效"有一部分是 k1 估计器人为引入（重要的诚实发现）。

**输出**：`data/phase4/e5_estimator.json`。

**成本**：低（只重算标量 $\tilde R$）。

---

### E6（P2，中）— 层可加性：扩到所有 MLP 层 + 求和 + cross-layer Fisher 质量

**原理**：Grosse 2023 是逐层算 IF 再求和，且反对"只看单层"，前提是 Hessian 跨层近似块对角，但从未实测 cross-layer Fisher 质量。补这个测量是独立贡献。

**计算**：
1. IF 从 layer-9 $W_2$ 扩到所有 12 层 MLP 的 $W_1,W_2$（每层独立 EK-FAC，复用 Phase 2），per-rollout 分数逐层相加得"全 MLP IF"。
2. 对比"全 MLP IF" vs "仅 layer-9 $W_2$"排序（Spearman/Jaccard）。
3. **cross-layer Fisher 质量（新测量）**：在 toy transformer（Phase 2 已有 toy + 暴力 Fisher 设施）上构造跨两层 $W_2$ 的联合子空间，暴力算联合 Fisher，测 cross-layer block 的 Frobenius 占比（类比 Phase 2 已测的 within-layer off-diagonal 45.6%）。
4. （可选）真实 GPT-Neo 上用随机投影估相邻层 cross-layer Fisher 质量代理。

**判据**：cross-layer 质量小（<10%）→ 逐层求和合理，论文可扩全 MLP 提升实证；大（>25%）→ 块对角在跨层也有不可忽略代价（诚实暴露新维度）。

**输出**：`data/phase4/e6_layer_additivity.json` + toy cross-layer 占比。

**成本**：全 MLP IF 约 12× 单层；toy cross-layer 便宜。

---

### E7（P2，便宜）— √KL 轨迹诊断

**原理**：Bai 2022 的 $R\propto\sqrt{\mathrm{KL}}$、Gao 2022 用 $d=\sqrt{\mathrm{KL}}$ 参数化过优化。在 31 个 checkpoint 上复现，给"θ\* 在轨迹什么位置、离最优多远"一个经验图像（支撑 E2 解释，也回应 Yide 之前'在 reward 最大处再看看'的直觉）。

**计算**：
1. 对 31 个 checkpoint，算评估集上 $\sqrt{D_\text{KL}(\pi_{\theta_t}\|\pi_0)}$（$\pi_0$=初始 GPT-Neo）与平均 reward，画 reward-vs-√KL，看是否近线性。
2. 标出 θ\*=step_0650 位置；C 阶段（过训练）是否偏离线性。
3. **不要**试图算 $D_\text{KL}(\pi_{\theta^*}\|\pi^*)$——需 $Z(x)$ 配分函数，不可解，跳过。

**判据**：θ\* 在仍上升的 √KL 曲线中段（非渐近平台）→ 经验上"策略仍在轨迹上、未到最优"，与 E2 非平稳、A2 失效一致（解释性证据，非定量距离）。

**输出**：`data/phase4/e7_sqrtkl.json` + `reports/phase4/e7_reward_vs_sqrtkl.png`。

**成本**：低（复用 checkpoint 评估）。

---

## 4. 跨实验决策逻辑

| 误差源 | 由谁量 | 转机条件 | 锤死条件 |
|---|---|---|---|
| 平稳性 $\|G(\theta^*)\|$ | E2 | 自然梯度步长 ≪ 训练单步 KL | ≳ 训练单步 KL |
| 曲率 $\Delta$（中心） | E3 | 4 目标 Spearman>0.95 且 $\|\tilde\Delta p\|/\|g_f\|\lesssim0.1$ | 任一目标 Spearman<0.9 或比值 ≳0.3 |
| within 方差构成 | E1, E5 | 大半是 reward 面 / 长度 / k1 人为 | 大半是 KL 度量上的真实策略次优 |
| 交叉验证 | E4 | A2-无关方法与 IF 高一致 | 低一致 |

**【修订 · 一阶可信域写进决策】** E3 的转机/锤死判定**必须先看 $\|\tilde\Delta p\|/\|g_f\|$**：该比值小（≲0.1）时一阶 Spearman 可直接采信；该比值大（≳0.3）时一阶结论不可信，结论以全阶 MINRES 为准。中间地带两者都报。

**综合结论模板**（按此写 Phase 4 report 结论段）：
- E3 转机 + E2 平稳性误差小 + E4 高一致 → **"尽管 ICC=0.892（A2 在 $y$ 层级不成立），$\Delta$ 修正与平稳性偏差对 per-rollout IF 排序的影响在 4 个评估目标上均可忽略（量化界见 E3）；论文的 $-\nabla f^\top F^{-1}\nabla\log\pi$ 排序是一个被验证的、一阶精确的局部归因。"** —— 把"PBRF 退而求其次的辩护"升级成"有误差预算的实测近似"。
- E3 锤死 → 诚实报告在哪些目标、哪些（很可能 E1 长尾 $A$ 大）rollout 上排序被 $\Delta$ 改变；给修正版排序作替代。
- 中间结果按目标/rollout 分层诚实报告，不合并。

---

## 5. 报告格式

在 `/root/rlhf-influence/reports/phase4_error_budget.md` 写报告，沿用 stop-and-report 规范：每个实验给（1）实际做了什么（2）输出文件路径+大小（3）每条验收判据的实测数值与转机/锤死判定（4）与计划偏差及原因（5）跳过/未完成明确列出。**关键产物**：E3 的 4×（Spearman, Jaccard, $\|\tilde\Delta p\|/\|g_f\|$, 修正项相对幅度分布）表 + split-half 收敛与去偏诊断 + 综合结论段（按 §4 模板）。

**【修订 · 统计自检清单，交报告前逐条确认】**
- [ ] 所有高维范数平方 / 二次型都用 split-half，无单批自积偏差。
- [ ] E2/E3 的 on-policy 采样确实 $y\sim\pi_{\theta^*}$，非复用训练 rollout。
- [ ] E3 中 $F$ 的 pseudo-label 与 $A$ 的 response 是同一次抽样的同一 $y$。
- [ ] $\tilde\Delta p$ 的两半余弦相似度足够高（收敛），否则加大 $N,K$ 重做。
- [ ] 全阶 solver 用 MINRES（对称不定），记录残差/迭代；未触发则说明未触发。
- [ ] 一阶 Spearman 的采信与否，明确依据 $\|\tilde\Delta p\|/\|g_f\|$ 的实测值。
- [ ] "距离最优""误差小"等解读与实测数字分开写。

> 实事求是：不可逆/昂贵步骤（全阶 MINRES、Bayesian-IF）只在边界情形触发并说明触发原因。
