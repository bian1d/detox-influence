# Fisher 逆全流程独立审查报告

审查对象:`src/ekfac/`(EK-FAC 因子、特征分解、IHVP、影响力打分)及
`src/phase4/`、`src/run_ekfac_production.py` 里所有用到 Fisher 逆的代码。

审查原则:**独立重验**。每个真值都用 numpy(`np.kron`/`np.linalg.inv`/
`np.linalg.eigh`)或闭式枚举从零算出,独立重新推导每个约定(vec 排序、响应
窗口切片、Λ 布局、阻尼),把项目代码当黑盒被测物。不复用现有测试的真值。

环境:base 老环境(torch 2.1.2+cu121 / numpy 1.26.3 / transformers 4.44.2 /
trl 0.9.6),与当初出 kill 结论的环境一致。GPU:RTX 4090 48G。

审查脚本(可交执行 agent 复跑):
- `tests/audit_1_dense_vs_ekfac.py` —— 稠密逆 vs EK-FAC(算法层 oracle + 真实 toy)
- `tests/audit_3_known_answer_e2e.py` —— 端到端已知答案(扁平 softmax,闭式真值)
- `tests/audit_3b_pseudolabel_offbyone.py` —— 错位 bug 机理三连证
- `tests/audit_3c_known_answer_corrected.py` —— 修复后端到端正控(收敛到真值)
- `tests/audit_4a_production_artifacts.py` —— 生产 artifact 直接复核(GPU)
- `tests/audit_4b_offbyone_damage.py` —— 错位 bug 在真实 GPT-Neo 上的损害
- `tests/audit_5_metric_scale.py` —— F/Δ 度量尺度 + 采样来源一致性(GPU)

---

## 一句话结论

EK-FAC 的逆代数(投影、阻尼、Kronecker 组装、特征基往返)**全部机器精度正确**;
但 **Fisher 的统计估计有一个真实的 bug:Stage 1 的 pseudo-label 与损失行错了
一位**,导致整张 Fisher 的 S 因子在真实模型上偏约 **378%**。这个 bug 不推翻
Phase 4 的 kill 结论(温度计仍远超阈值,且尺度修正只会让 kill 更确定),但它
**让生产的"最有影响力 rollout"前 100 名约 1/3 换人、每条影响力值偏几十个
百分点**,且 Fisher 在数学上不是它声称估计的量。建议修复(一行)后重算因子。

---

## 第一刀:稠密直接求逆 vs EK-FAC 近似逆 —— 通过

**怎么验**:在小到能显式造出整张 Fisher 的设置上,用 `np.linalg.inv` 对稠密
Fisher 求绝对正确的逆,对照 EK-FAC 的近似逆。分两层:

Rung A(纯线性代数 oracle,合成相关数据,无模型):

| 门 | 验的是什么 | 结果 |
|---|---|---|
| A1 | 项目特征分解能对角化我的 A/S | rel err 9e-16 ✓ |
| A2 | 我独立推的 Λ 布局恒等(稠密对角 == 直接公式) | 3e-15 ✓ |
| A3 | `inverse_hvp_additive` == 我的 numpy oracle | 4e-16 ✓ |
| A4 | `reconstruct_F_inv_dense` == 我的 oracle 稠密矩阵 | 0.0 ✓ |
| A5 | 两级阻尼只加一次(含 floor 分支) | 1e-14 ✓ |
| A6 | 特征基往返 == 恒等 | 7e-16 ✓ |

Rung B(真实 toy transformer,被动同抽样捕获):

| 门 | 验的是什么 | 结果 |
|---|---|---|
| B3 | hook 语义:所有位置 Σ δ⊗m == autograd 梯度 | 0.0 ✓ |
| B0 | token 记账(我的窗口 == 管线 n_tok) | 385==385 ✓ |
| B1 | A,S == 我独立稠密累积(同抽样) | 7e-16 ✓ |
| B2 | Λ == 我稠密 diag(Uᵀ F U)(同抽样) | 3e-15 ✓ |

**近似误差(信息项,非门)**:
- 相关合成数据:EK-FAC 逆 vs 稠密逆 Frobenius rel 0.181,off-diag Kronecker 质量 0.443。
- 独立合成数据(总体即 Kronecker):0.0095 / 0.0405(残留仅有限样本噪声)。
- 真实 toy transformer:Frobenius rel **0.075**,作用在随机梯度上的 rel 误差中位 **0.071**、
  max 0.113;off-diag Kronecker 质量 0.428。影响力排序(稠密逆 vs EK-FAC 逆)Spearman **0.994**。

**判读**:逆代数本身正确无误。EK-FAC 与真正稠密逆的差异(7-8% Frobenius)**完全
由 Kronecker 独立性假设解释**(off-diag 质量 ~44%,与 CLAUDE.md 记录、Grosse
2023/MDA 公布范围一致),不是实现 bug;排序层面差异极小(0.994)。**这一刀通过。**

---

## 第二刀:阻尼加在哪、加几次 —— 通过

独立追踪 `inverse_hvp` 的 `denom = max(Λ + 0.1·Λ̄, 1e-5)`:构造特征方向
`g = outer(Q_S[:,j], Q_A[:,i])`,它经过 IHVP 必须精确等于 `g / max(Λ[i,j]+α·mean,
floor)`。多除/少除一次都会立刻暴露。

- 合成数据(A5,含人为压到 floor 以下的项以触发 clamp 分支):worst rel 1e-14 ✓
- **真实生产 Q_A/Q_S/Λ**(4a-C7,随机真实特征方向):worst rel 1e-14 ✓

**判读**:阻尼**恰好加一次**,加在特征值上,floor 与 alpha·mean 两级都正确,clamp
分支正确。生产 `denom` 与 Stage 3 缓存 `g_scaled_z` 不存在重复除阻尼(另见 4a-C3
精确为 0,p == IHVP(g))。**这一刀通过。**

---

## 第三刀:特征基投影/逆投影是否互逆 —— 通过

A6 直接验 `Q_S (Q_Sᵀ g Q_A) Q_Aᵀ == g`,rel 7e-16(机器精度)。Q 正交性在合成
(A1)与真实生产因子(4a-C1:‖Q_A Q_Aᵀ−I‖=2e-15)上都机器精度。**这一刀通过。**

---

## 第四刀:F 与 Δ 的采样/度量一致性 —— 发现两个问题

### 4.1 度量尺度:逐 token Fisher vs 逐样本分数(实测 10–19 倍)

**机理**:`accumulate_AS` 内部用 `reduction="sum"` 反传,再把逐 token 外积**除以
token 总数** `n_tok` —— 估的是**逐 token Fisher** `F_tok = E_t[g_t g_tᵀ]`。而
`compute_per_sample_grad` 和 Δ 里的 s(`hvp_logpi`)都用 `reduction="mean"`,是
**逐样本**(对 R 个 token 平均)梯度。影响函数要逆的是总训练损失(逐样本损失
平均)的 Hessian,其 Fisher 近似应是**逐样本分数** Fisher `F_samp = E_i[s_i s_iᵀ]`。
两者约差一个响应长度 R 的因子。

**实测**(audit 5,真实 θ\* 上一池 320 条 on-policy 样本,mean R=22):

| 方向 | q_tok/q_samp(尺度比) |
|---|---|
| p_seq | 16.3 |
| p_toxic_C1 | 19.0 |
| g_seq | 10.7 |
| g_toxic_C1 | 11.2 |
| random | 19.3 |

即生产 F(逐 token)是与 Δ 同度量的逐样本 Fisher 的约 **10–19 倍**(比 R=22 略小,
因为样本内 token 部分相关)。生产 EK-FAC 前向 Fisher `vᵀF v` 与经验逐 token
Fisher 同数量级(比值 0.18–2.1),确认生产 F 确属逐 token 尺度。

**后果(方向务必看清)**:
- **对影响力排序:基本无害。** I = −g_fᵀ F⁻¹ s_m 对 F 的全局缩放不变;尺度比在
  各方向 10–19(2 倍以内变动),排序近似保持。Phase 2 生产排序安全。
- **对温度计(Phase 4/5 kill 决策变量):** 温度计 ‖Δ̃ p‖/‖g_f‖ 把逐样本尺度的 Δ
  和逐 token 尺度的 F(经 p=F⁻¹g_f)混用。设 F_tok = c·F_samp(c≈16),则 p 偏
  **小** c 倍,故生产温度计比"度量自洽的温度计"**小** c 倍。即度量自洽的温度计
  ≈ c × 报告值 ≈ 34/12/4/12 的约 16 倍(~550/190/64/190)。**方向是让 kill 更
  确定,不是威胁它**;最小的 C2(4→~64)也远在 0.3 之上,无目标接近阈值。
  > 注:`audit_5_metric_scale.py` 我初版 M3 散文把除法方向写反了(写成 34/16),
  > 已在脚本中更正。实测比值无误,只是那段叙述错了。Δ 还含 HVP 项(缩放方式不同),
  > 故上面是数量级估计,不是精确除法。

**建议**:统一一个 reduction 约定让 F 与 s 同度量(把 F 改成逐样本分数 Fisher,
或显式声明 F 按逐 token 设计、温度计绝对值据此解读)。两种都不改变任何结论;但
现状下"温度计=34 即 Δ 是 F 的 34 倍"这种字面读法在度量上不成立。

### 4.2 采样来源:F 用历史 rollout,Δ 用 on-policy(偏离 spec 意图)

- 生产 **F**(`run_ekfac_production.py`)在 **20,832 条历史训练 rollout**(step 0..650,
  相对 θ\* 是 off-policy 的输入上下文)+ θ\* pseudo-label 上累积。
- **Δ**(`run_phase4_e3.py`)在**全新 on-policy 池**(θ\* 现采 200×16)+ 真实 token
  (s/HVP)+ 真实奖励(A)上算。

`phase4.md` §1 E3【修订】明确要求"F 的 pseudo-label 与 A 的 response 是同一次
on-policy 抽样的同一个 y"。生产管线**没有**这么做:它复用了为 Phase 2 排序而建的
大 F,另起一池建 Δ。这是对 spec 意图的偏离;两者都是合法 Fisher,影响的是"估的
是哪个分布下的 Fisher",由排序稳健性(4b:Spearman 0.97–0.99)与 Phase 1 ICC 工作
间接界定为二阶效应。**记录在案,建议复盘是否要按 spec 用同一 on-policy 池重建 F。**

---

## 第五刀:端到端"已知答案" —— 抓到一个真实 bug

### 5.1 构造

扁平 softmax 模型(无注意力、无跨位置流):token → 冻结 embedding → c_proj(唯一
可训层)→ 冻结 readout → logits。每个位置的标签分布独立,故逐 token Fisher 有闭式:
`Σ_t = headᵀ(diag(p_t) − p_t p_tᵀ)head`,`F_exact = mean_t kron(Σ_t, m_t m_tᵀ)`。
影响力真值 = 全 numpy 的 `−vec(g_eval)ᵀ inv(F_exact+λI) vec(s_m)`。把项目完整管线
当黑盒跑,对照真值。误差分解:`|I_管线 − I_经验稠密|`=EK-FAC 结构误差;
`|I_经验稠密 − I_真值|`=pseudo-label 偏置 + 有限样本。

### 5.2 抓到的 bug:Stage 1 pseudo-label 错位一位

**机理(独立逐索引推导)**:`factors._accumulate_one` 里 `pseudo[u] ~ softmax(logits[u])`;
经 shift-by-one 对齐后,损失第 t 行(用 `logits[t]`,即位置 t+1 的预测分布)配到的
标签是 `pseudo[t+1] ~ softmax(logits[t+1])`,也就是**下一个位置**的条件分布。
而 Fisher 要求第 t 行标签从它自己那行的 `softmax(logits[t])=pseudo[t]` 采。

**三连独立证**:

- **N1(扁平模型,闭式偏置极限)**:我先解析推出错位采样会收敛到的偏置 S 极限
  `diag(q_next)+p pᵀ−p q_nextᵀ−q_next pᵀ`(投影到 head 上),再跑管线 —— 管线 S
  落在偏置极限上(rel **0.008**),离正确极限 **0.258**。坐实。
- **N2(真实 toy transformer,我自己的 hook+采样)**:管线 S 匹配"下一行采样"
  (0.052,≈噪声),不匹配"行对齐采样"(0.194,≈两者间距 0.203)。坐实。
- **N3(toy 损害)**:buggy vs 修正影响力 rel RMS 0.072,Spearman 0.998。

**端到端可见**:audit 3 的 v1(Kronecker-exact)"最终精度"门 **FAIL** —— 管线影响力
总误差 0.42、Spearman 0.95,且 n=400→4000 几乎不降(0.47→0.42,**偏置主导**而非
MC 主导)。即端到端已知答案测试**没能收敛到正确暴力真值**,根因就是错位。

**正控闭环(audit 3c,修复后)**:把 `response_labels` 改成 `pseudo[P-1:T-1]`(行对齐),
其余不变 —— v1 总误差 0.42→**0.029**,Spearman 0.95→**1.0000**,且随样本 0.088→0.029
收敛(偏置消失)。**证明一行修复后管线确实正确**;v2 残留 0.15 是 EK-FAC 真实的
非 Kronecker 近似误差(非 bug),Spearman 0.986 稳定。

### 5.3 真实规模损害(audit 4b,GPT-Neo θ\*,同子集 2604 条苹果对苹果)

| 量 | 值 | 说明 |
|---|---|---|
| ‖A_bug − A_corr‖/‖A_corr‖ | **0.0000** | A 不依赖标签,确认差异只在 S/Λ |
| ‖S_bug − S_corr‖/‖S_corr‖ | **3.78** | S 被错位污染约 378% |
| ‖p_bug − p_corr‖/‖p_corr‖ | 0.25 | IHVP 方向 cos 0.97,‖p‖ 变 9% |
| I_seq 排序 Spearman | 0.986 | top-50/100 Jaccard 0.69/0.67 |
| I_toxic 排序 Spearman | 0.970 | top-50/100 Jaccard 0.69/0.79 |
| 逐条幅度偏移 | 中位 23–28%,p90 60–100% | |

**为何 S 偏这么大**:GPT-Neo 的 next-token 分布很尖锐。正确采样下标签≈argmax,
梯度 `p_t−e_y≈0`(自信模型 → Fisher 本该很小);错位把下一位置 token 当标签,
产生大的虚假"双热"梯度,**在模型最自信处把 Fisher 系统性灌大**。这也解释了为何
排序还较稳(阻尼 + 主特征结构吸收了大部分污染),但 top-k 成员与逐条幅度受实质影响。

### 5.4 为何现有 toy 门没抓到(你担心的"自己改自己的卷子")

`tests/test_eigen_toy.py::test_lambda_matches_kronecker_diagonal` 用
`compute_per_token_fisher_beta` 当"真值",但**那个函数和 `fit_lambda` 用同一个
`sample_pseudo_labels`+`pseudo[plen:]`**,两边带同样的错位 → 自洽地相等、门通过、
但都错。我审计 1 的 B2(fit_lambda == 稠密对角)也复用了管线实际捕获的 m/δ,同样
只验**内部自洽**。只有 audit 3/3b/3c/4b 这种"不复用管线标签、从行对齐独立重算
真值"的路子才能暴露它。这正是 EK-FAC 这类自洽全绿陷阱的典型。

---

## 生产 artifact 直接复核(audit 4a,12/12 通过)

把磁盘上的生产张量当不可信输入,fp64 从零重导每条一致性关系:
- C1 Q 正交性 2e-15;C2 A/S 特征重构 5e-15;C7 真实特征方向上两级阻尼 1e-14。
- C3 p_seq/p_toxic{C1,C2,C3} == 现算 inverse_hvp(g) **精确 0.0**。
- C4 抽样 25 条 rollout,用独立梯度路径复算 I_seq/I_toxic == 存盘值,worst rel 3e-15。
- C5 g_toxic_C1 从文本重算 == 存盘 0.0;C6 自影响 5/5 负号、top-5。

**判读**:生产管线对自己**完全自洽** —— 存的 p/I 就是 buggy 代码该算出的值。错位
bug 不是缓存错配,而是 Fisher **定义**本身的错,贯穿一致。

---

## 修复

两处一行改动(`pseudo[P:]` → `pseudo[P-1:T-1]`):

- `src/ekfac/factors.py` `_accumulate_one`:`response_labels=pseudo[P:]` → `pseudo[P - 1 : T - 1]`
- `src/ekfac/eigen.py` `_fit_lambda_one`:同改

(`tests/test_eigen_toy.py::compute_per_token_fisher_beta` 也有同样的 `pseudo[plen:]`,
需同步改,否则那条"真值"门继续 bug-to-bug 自洽通过。)

`tests/audit_4b_offbyone_damage.py` 里的 `corrected_accumulate_AS` /
`corrected_fit_lambda` 已是可直接参考的正确实现;audit 3c 已验证修复后端到端收敛
到暴力真值(Spearman 1.0)。

**修复代价**:作废 Phase 2/4/5 所有因子 artifact(A/S/Q/Λ/p/I/缓存),需重算
`run_ekfac_production.py` 及下游。**这是研究决策(重算成本 vs 结论是否变),按
CLAUDE.md 工作约定我只 surface、不擅自改生产代码与 locked-in 决策。**

**对已发表结论的影响(基于实测,不带预期)**:
- Phase 4 kill(温度计 ≫0.3):**稳健**。p 仅变 25%,且 4.1 的度量修正只会让温度计
  更大;kill 不变。
- Phase 2 生产**排序**:Spearman 0.97–0.99 存活,但 **top-100 约 1/3 换人**、逐条
  影响力值偏几十个百分点。若论文要点名"最有影响力的具体 rollout",这部分需用
  修复后的因子重算。
- 所有"机器精度正确"的声明对**逆代数**成立;对 **Fisher 统计估计**不成立(它没在
  估它声称的量)。

---

## 五刀总览

| 刀 | 结论 |
|---|---|
| 1 稠密逆 vs EK-FAC | 通过 —— 逆代数正确,近似误差在可解释范围(7-8% Frob / 0.99 排序) |
| 2 阻尼加几次 | 通过 —— 恰一次,机器精度,含 floor 分支与真实因子 |
| 3 投影往返 | 通过 —— 机器精度恒等 |
| 4 F/Δ 一致性 | **两问题** —— 度量尺度逐 token vs 逐样本 10-19×(排序无害、kill 更稳);采样来源偏离 spec |
| 5 端到端已知答案 | **抓到 bug** —— Stage 1 pseudo-label 错位一位,S 偏 378%,排序 top-1/3 换人 |
