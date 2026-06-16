# 修复 + 重做方案(Fisher 错位修复后的 GPT-Neo 全量重做)

> 给执行 agent:本文件是自包含落地方案。先读 `CLAUDE.md`、`CONTEXT.md`、
> `reports/fisher_inverse_independent_audit.md`(Fisher 逆独立审查,错位 bug 的
> 证据与机理)。所有数学用文字写(Fisher 逆 = inverse-F,score 梯度 = s_m,
> 评估方向梯度 = g_f,污染算子 = Delta,温度计 = ‖Delta·p‖/‖g_f‖)。
> 每个 Block 末尾按 CLAUDE.md 的 stop-and-report 协议停下等人工 review。

---

## 0. 三个先要拍板的研究判断(设计依赖它们)

**判断 A —— 前 10% 分析是"探索性刻画 + kill 的交叉检验",不是被验证的归因。**
项目已对 GPT-Neo 下 kill 结论(温度计 4-34,Delta 把排名重排)。在已 kill 的模型上
说"这是最有影响力的 10%"自相矛盾。但它仍有研究价值,且要这样做:
- 同时用**一阶影响力**(论文公式)和 **Delta 修正后影响力**两套排名各取前 10%;
- 关键测量不是整体 Spearman,而是**前 10% 名单在两套排名间的 Jaccard 稳定性**。
- 含义:高温度(整体重排)**不一定**摧毁"前 10% vs 其余"这种粗粒度二分——重排
  可能集中在噪声中段。若前 10% 在两套排名间稳定、且确实是最毒/最极端的 →
  说明"精排归因失败但粗粒度高/低归因存活",是比"一刀切 kill"更细的结论;
  若前 10% 不可解释或两套之间剧烈漂移 → 强化 kill。**两种结果都有价值,不要带
  预期。**

**判断 B —— 既然要重建因子,顺手修两个审查发现的度量问题(低成本、提可信度)。**
1. 【已定:用现采 on-policy】生产 inverse-F 当年建在**历史训练 rollout**(相对最优是
   off-policy 的上下文)上;改建在**现采的 on-policy 样本**(从 step_0650 当场采)上,
   使 F 和 Delta 在同一测度。符合 spec §1 E3、与 Phase 5 一致。改用
   `phase5/sample_rollouts.py::generate_factor_rollouts` 的同款做法。
   注意:被打分/归因的对象**仍是 20,832 条历史训练 rollout**(那才是要归因的 z_m);
   只有"建 Fisher 的数据"换成 on-policy。两者本就该不同(Fisher 要 on-policy 测度,
   归因对象是历史数据)。
2. 逐 token vs 逐样本度量(审查实测 F 比逐样本分数 Fisher 大 10-19 倍):**不改 F 的
   逐 token 约定**(它是 EK-FAC/MDA 标准,且排序对全局缩放不变),但温度计要在
   报告里同时给"原始值"和"度量自洽值(约 ×16)",避免把"温度计=34"字面读成
   "Delta 是 F 的 34 倍"。

**判断 C —— GPT-Neo 重跑是"确认 kill 在修复后仍成立 + 产出干净数字 + 解锁前 10%
分析",不是推翻 kill 或重新发现 kill。** 审查已证明 p 只变 25%、度量修正让温度计
更大,故预期 kill 不变。重跑是为了把带 bug 的旧数字换成干净数字,并让前 10% 分析
建立在正确因子上。不要把它写成新结论。

> 另:Phase 5 的 OLMo-SFT 温度计(记忆里 1031/1096,DEATH)是用**同一个带 bug 的**
> `accumulate_AS`/`fit_lambda` 算的,也受错位影响,且 1031 vs GPT-Neo 的 34 差约 30 倍
> (接近一个 R 尺度或 bug 尺度),**修复后必须重测**才能下"A2 失效非 PPO 特有"的定论。
> 见 Block C。

---

## Block A —— 模型无关的基础修复(最先做,GPT-Neo 与 OLMo 都受益)

### A1. 修复 Fisher 错位 + 把修复收敛到一个 helper

**根因**:loss 第 t 行(用 logits[t])配到的标签来自 softmax(logits[t+1])——下一个
位置的条件分布;Fisher 要求来自本行 softmax(logits[t])。

**改法(altitude:一处 helper,杜绝再漂移)**:在 `src/ekfac/factors.py` 加

```python
def sample_row_aligned_labels(
    logits_row: torch.Tensor,          # (T, V) 单条序列的 logits
    prompt_ids: torch.Tensor, response_ids: torch.Tensor,
    *, generator: torch.Generator, ignore_index: int,
) -> torch.Tensor:
    """行对齐的 Fisher pseudo-label:loss 行 t 的标签 ~ softmax(logits[t]).
    返回 (P+R,) 的 masked labels。修复点集中在此,所有 Stage-1 调用它。"""
    P = int(prompt_ids.shape[0]); T = P + int(response_ids.shape[0])
    pseudo = sample_pseudo_labels(logits_row, generator=generator)   # (T,)
    _, labels = build_input_and_labels(
        prompt_ids, response_ids,
        response_labels=pseudo[P - 1 : T - 1],   # 关键:P-1..T-2,不是 P..T-1
        ignore_index=ignore_index,
    )
    return labels
```

把 `factors._accumulate_one` 与 `eigen._fit_lambda_one` 里那段"sample_pseudo_labels
+ build_input_and_labels(response_labels=pseudo[P:])"替换成调用上面的 helper。

**找全所有点**:执行前先 `grep -rn "pseudo\[" src/ tests/` 与
`grep -rn "sample_pseudo_labels\|response_labels=" src/ tests/`,确保无遗漏
(已知:`factors.py`、`eigen.py`、`tests/test_eigen_toy.py::compute_per_token_fisher_beta`)。

### A2. 重写"自己改自己卷子"的测试

`tests/test_eigen_toy.py::compute_per_token_fisher_beta` 当前与 `fit_lambda` 共用同一
错位,所以 bug-to-bug 自洽通过。改成**独立真值**:

- 该函数也改用行对齐采样(`pseudo[P-1:T-1]`),这样它成为对齐后的稠密真值;
- 但更重要:把审查脚本里两个独立真值测试**升级为常驻回归测试**(它们不复用管线标签):
  - `tests/audit_3c_known_answer_corrected.py` → 收敛到闭式真值、Spearman 1.0;
  - `tests/audit_1_dense_vs_ekfac.py` → 逆代数机器精度。
  把这两个加进 pytest(或 CI 脚本),作为"Fisher 估计正确性"的硬门。
- 删除或标注 `audit_3_known_answer_e2e.py` 里那个**故意 FAIL** 的 v1 门——它在 bug
  存在时 FAIL(0.42、Spearman 0.95),修复后应自动变绿(总误差→0.03、Spearman 1.0);
  保留它作为"bug 回归哨兵"也可(注释说明 FAIL=bug 复现)。

**验收门 A**:`audit_1`、`audit_3c` 全绿;`audit_3`(修复后)v1 final-accuracy 门转绿
(总误差 < 0.05、Spearman > 0.99);`pytest tests/` 绿。

---

## Block B —— 删 184G 缓存,改现场计算 + 多 MLP 层 + 稳健方向

### B1. 删缓存,统一成"一遍现场打分"

当前 `run_ekfac_production.py` 把每条 rollout 的 `g_scaled_z` 写盘(~184G),
`phase4/e3_delta.py::cache_dot_scores / corrected_scores_from_cache` 读盘。**全删**。

**新结构(一遍 backward 服务所有目标、所有层、一阶+修正)**:

```
预备(每个评估目标 f、每层 L 各一次,便宜):
  p[f][L] = inverse-F_L 作用于 g_f[L]                  # 一阶方向
  q[f][L] = inverse-F_L 作用于 (Delta~_L 作用于 p[f][L])  # Delta 修正方向(若该层做修正)

打分(一遍过 20,832 条 rollout):
  for 每条 rollout z_m:
     一次 forward+backward,hook 抓所有 MLP 层的 (m, delta)
     for 每层 L: s_m[L] = 该层 mean-reduced score
     for 每个目标 f:
        I_firstorder[f][m]  = sum_L  -<p[f][L], s_m[L]>
        I_corrected[f][m]   = sum_L  -<p[f][L]+q[f][L], s_m[L]>   # 若该目标要修正
  不落盘任何 per-rollout 张量;只存最终 I 向量(每个目标一个长度-N 数组)。
```

要点:打分不缓存中间量,内存常数级;新增评估目标需重跑这一遍(用户已接受)。

**改动文件**:重写 `src/run_ekfac_production.py`(或新建 `src/run_influence_gptneo.py`);
删 `CACHE_DIR` 相关代码;`phase4/e3_delta.py` 的 cache_dot_scores 路径改为接收
现场算的 `s_m[L]` 流(或把打分循环搬进一个公共 `score_rollouts_streaming()`,E3 复用)。

### B2. 多 MLP 层 EK-FAC(要求 2)

**层集合**:GPT-Neo 12 层,每层 MLP 的 `c_fc`(W_1,d_model→d_mlp)与
`c_proj`(W_2,d_mlp→d_model),共 24 个 nn.Linear。**只 MLP,不碰 attention 的
QKVO**(softmax 让 Kronecker 近似变脏,Grosse 2023 惯例)。

**新增多层版本(altitude:推广机制,而非 24 次单层 pass)**:

```python
# ekfac/hooks.py
@contextmanager
def capture_mlp_layers(layers: dict[str, nn.Linear]) -> Iterator[dict[str, CProjCache]]:
    """一次性 hook 多个线性层;一遍 forward+backward 填满每个 cache。"""

# ekfac/factors.py
def accumulate_AS_multi(
    model, layers: dict[str, nn.Linear], rollouts, cfg, *, device,
) -> dict[str, tuple[torch.Tensor, torch.Tensor, int]]:
    """一遍过 rollouts,返回每层 (A, S, n_tok)。行对齐 pseudo-label(走 helper)。
    一次 forward → 采一次行对齐标签 → 一次 backward → 各层 hook 各自累积。"""

# ekfac/eigen.py
def fit_lambda_multi(
    model, layers, rollouts, cfg, *, Qs: dict[str, tuple[Tensor, Tensor]], device,
) -> dict[str, tuple[torch.Tensor, int]]:
    """第二遍(行对齐,seed+1);每层在自己的 (Q_A, Q_S) 基里拟合 Lambda。"""
```

**内存**:fp64 下每个 c_proj 因子约 173MB,c_fc 约 173MB,24 层约 4.2GB;48G GPU 放得下,
但建议因子主存在 CPU、按层搬上 GPU 算 IHVP。eigh 在 3072×3072 fp64 上是秒级。

**层间求和的正确性**:总参数 inverse-F 跨层近似块对角,逐层 I 相加 = 块对角 inverse-F
影响力。所有层统一用逐 token 约定,层间相对权重一致,求和合法(见判断 B-2)。

**新测量(phase4.md E6,研究贡献)**:对比"仅 layer-9 W_2 影响力" vs "全 MLP 求和影响力"
的 Spearman/Jaccard。一致 → 单层够代表、当年的 kill 不是选层 artifact;不一致 → 单层
不够,需报告。注意:加更多层**不修复** A2/Delta 问题(每层各有自己的 Delta),它回答的是
子空间代表性,不是救 kill。

### B3. 评估方向改 100 条平均稳健版(要求 1)

当前 f_toxic 用单模板(x_eval, y_toxic),偶然性大。改成 Phase 5 已有的 `build_g_f`:

- 复用 `src/phase5/toxicity_direction.py::build_g_f` + `load_lee_pairs`,**移植回 GPT-Neo**
  (Phase 5 是 OLMo 版,逻辑相同,只换 model/tokenizer/层)。
- 方向 = 约 100 条该类毒性的 (prompt, 毒 response) 与配对 (prompt, 中性 response) 各算
  mean-reduced score,**基线相减**(平均毒 − 平均中性)去掉话题、只留毒性;同时保留
  非相减的"纯毒"版做对照。每层各自得一个 g_f[L](一次 backward 出全层)。
- 报告:相减后范数 < 纯毒范数、cos(毒,中性) 高(说明基线相减确实去掉了共同话题方向)
  —— Phase 5 已有这套 sanity 输出,照搬。
- f_seq 已是 100 prompt × 8 续写的平均,保留;但也按多层重算其每层梯度。

**数据**:Lee et al. 的毒/中性配对(`references/DPO-detoxify/`,Phase 5 用过 24576 条)。
GPT-Neo 上各类取约 100 条配对。

**验收门 B**:多层 `accumulate_AS_multi` 在 toy transformer 上与"逐层单独跑 `accumulate_AS`"
逐层逐元素一致(机器精度,证明多层 hook 没串扰);现场打分对单条 rollout 复算出的 I
与"老式单层 + 逆"在 layer-9 上一致(到 fp32 量级);build_g_f 的相减 sanity 通过。

---

## Block C —— GPT-Neo 全量重跑 + 前 10% 分析(用修复后的因子)

按顺序、每步 stop-and-report:

**C1. 重建 inverse-F 因子(on-policy,修复后,单层 layer-9 先行)。**
用 `generate_factor_rollouts` 现采 on-policy 样本(token 预算对齐当年 ~60 万响应 token),
跑修复后的 `accumulate_AS` + `eigendecompose` + `fit_lambda`。先单层 layer-9 W_2,
确认与审查 4b 的 corrected 因子一致(S 与旧带 bug 的差约 378%、与 corrected 一致)。

**C2. 确认 kill 在修复后仍成立。**
重算温度计(4 个目标:f_seq + 3 个稳健毒性方向)与一阶/修正排名 Spearman。
预期:温度计仍远超 0.3(p 只变 25%、度量修正让其更大)。报告"原始温度计"与
"度量自洽温度计(约 ×16)"两栏。把新数字与旧 4-34 对照,解释差异来源。

**C3. 稳健方向(B3)+ 全 MLP(B2)上重算。**
用 100 条平均稳健方向重算温度计与排名;再扩到全 24 个 MLP 层,报告 E6 的
单层 vs 全 MLP 一致性。

**C4. 前 10% prompt 分析(要求 3,判断 A 的框架)。【已定:稳健毒性方向 + f_seq 都做】**
- 对**稳健毒性方向**和 **f_seq** 两个目标,各取一阶排名前 10% 与 Delta 修正排名前 10%;
- 三类稳定性测量:
  1. **一阶 vs 修正** 前 10% 的 Jaccard(同一目标内,Delta 把名单换了多少)——核心量;
  2. **稳健毒性方向 vs f_seq** 前 10% 的 Jaccard(两个目标是否指向同一批 rollout);
  3. 前 10% 的画像:奖励/RoBERTa 毒性分布、长度、所属 PPO step(早 step = 去毒前 = 更毒)、文本抽样;
- 与 bottom-10% 和随机基线对比三类画像;
- 诚实判读(判断 A):前 10% 是否确实最毒/最极端 + 在"一阶↔修正"和"毒性↔f_seq"两个维度
  上稳不稳,据此说"粗粒度高/低归因是否在精排 kill 下仍存活、是否目标无关"。

**输出**:`data/influence_gptneo_fixed/`(因子、I 向量、温度计 json)、
`reports/gptneo_fixed_redo.md`(温度计表、单层 vs 全 MLP、前 10% 分析表)。

---

## Block D —— OLMo(把同一套修复带过去)

GPT-Neo 跑通、修复被证明后,再处理 OLMo:
- Phase 5 的 `accumulate_AS`/`fit_lambda` 是同一份共享代码,Block A 的修复**自动**作用于它;
- **必须重测** OLMo-SFT 的 DEATH 温度计(1031/1096 受 bug 影响,且与 GPT-Neo 的 34 差约
  30 倍,要查清多少是 bug、多少是度量尺度、多少是真信号);
- 稳健方向机制 OLMo 已有(`build_g_f`),全 MLP 机制从 Block B 平移(OLMo 层名是
  `model.layers.{i}.mlp.{gate_proj,up_proj,down_proj}`,注意 OLMo MLP 是 3 个矩阵的
  gated 结构,需各自加 hook;gate/up 输入同源、down 输入是激活后)。
- 这才是回答"A2 失效是 PPO 特有还是方法死穴"的干净版本。

---

## 顺序建议(item 6)

```
Block A  修复 + 测试重写(模型无关,最先)         —— 半天,解锁一切
   │
Block B  删缓存 + 多 MLP + 稳健方向(基础设施)     —— 1 天,toy 上验收
   │
Block C  GPT-Neo 全量重跑 + 前 10% 分析            —— 用户熟悉、已 kill、快确认
   │
Block D  OLMo 重测(同代码平移)                    —— 回答 PPO 特异性
```

为何 GPT-Neo 在 OLMo 前:(1) 修复必须先在熟悉、已 kill 的模型上证明能复现/确认旧结论;
(2) OLMo 的 Phase 5 数字也受同一 bug 影响,不修不值得深挖;(3) 用户对 GPT-Neo 熟,
前 10% 这种需要"看文本、判断毒不毒"的探索性分析在熟悉模型上更靠谱。

## 全局验收纪律(CLAUDE.md)

- 每个 Block 末尾 stop-and-report:做了什么、输出文件+大小、每条验收门的实测值+通过否、
  偏离与原因、跳过项。
- 任何门失败**不准静默降标准**:报实测 vs 阈值、已试过什么、三个假设、能区分假设的证据。
- 不缓存中间量(用户决定);不碰 attention;只 MLP;行对齐 pseudo-label 走唯一 helper。
```
