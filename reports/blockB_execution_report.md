# Block B 执行报告 — 删缓存改流式 + 多 MLP 层 EK-FAC + 100 条平均稳健方向

> 配套独立 sub-agent review 见本报告末尾(独立重推真值,非复跑作者测试)。

## 做了什么

把单层 EK-FAC 流程推广到所有 MLP 层(一遍反传抓全层)、把 184G 磁盘缓存换成
现场流式打分、把单模板毒性方向换成 100 条 Lee 配对、基线相减的稳健方向。只 MLP,
不碰 attention 的 QKVO(softmax 让 Kronecker 近似变脏,Grosse 2023 惯例)。
错位修复经唯一 helper `sample_row_aligned_labels` 自动被多层路径继承。

## 新增 / 改动

- `src/ekfac/hooks.py`:新增 `capture_mlp_layers`(一次 hook 多层,纯观察者)。
- `src/ekfac/multilayer.py`(新):`accumulate_AS_multi` / `fit_lambda_multi`
  (多层 Stage 1A/1B,行对齐 pseudo-label)、`per_sample_grads_multi`
  (一次 `autograd.grad` 出全层 score,真实 token)、`influence_streaming`
  (缓存free 一遍打分,所有目标所有层,KL-RL/监督符号可选)。
- `src/eval_directions.py`(新):`build_g_f_multi`(多层、基线相减稳健毒性方向)。
- `src/downcast_cache.py`:**删除**(纯缓存工具,无人 import)。
- 新增门:`tests/test_multilayer_consistency.py`、`tests/test_streaming_influence.py`、
  `tests/test_robust_directions_gptneo.py`。

## 验收门(全绿)

| 门 | 验的是什么 | 结果 |
|---|---|---|
| B-1 多层==单层 | 4 个 MLP 层(c_fc=W_1, c_proj=W_2 两种朝向)的 A/S/Λ,多层一遍 vs 逐层单独算 | **逐元素 0.0e+00**(证明 hook 无串扰) |
| B-1 score 多层==单层 | `per_sample_grads_multi` vs `compute_per_sample_grad` 逐层 | **0.0e+00** |
| B-2 流式==直算 | `influence_streaming`(层求和)vs 逐层逐条直算参照,2 目标 | **0.0e+00**(rel 0) |
| B-2 符号 | KL-RL(−1)与监督(+1)两种约定 | 均 0.0e+00 |
| B-3 稳健方向 | GPT-Neo 30 条 Lee 配对,基线相减 sanity | **3/3 PASS** |

B-3 细节(GPT-Neo 真实模型,30 配对,24 个 MLP 层):
- cos(毒,中性) > 0:**24/24 层**(毒和中性梯度共享话题方向)。
- 相减后范数 < 纯毒范数:**24/24 层**(基线相减确实去掉共同话题分量)。
- 主层 layer-9 c_proj:cos=+0.623,‖相减‖/‖纯毒‖=0.804(去掉约 20% 共同话题质量)。

## 删缓存的范围说明(诚实标注,需你确认)

- **已删**:`downcast_cache.py`(纯缓存工具)。新影响力路径 `influence_streaming`
  **构造上就不落盘**,现场重算 score 现场点积。
- **本 Block 未动、留待 Block C 删**:旧单层缓存写盘驱动 `run_ekfac_production.py`、
  以及 phase4 的缓存读取器(`phase4/e3_delta.py::cache_dot_scores` /
  `corrected_scores_from_cache`、`run_phase4_e3.py` / `run_phase4_e4.py` /
  `score_extra_eval_targets.py` 里的缓存读取)。
- **为什么留到 Block C**:这些都是"驱动脚本",改写它们和**实际跑它们**是一个耦合
  动作(改完必须在 GPT-Neo 上跑一遍才知道对不对)。Block C 会写新的缓存free 驱动
  `run_influence_gptneo.py` 取代 `run_ekfac_production.py`,并在重跑 E3/E4 时把那几个
  缓存读取器换成 `influence_streaming`。在 Block B 里盲改这些驱动会留下"改了但没跑过"
  的未验证代码,违背"不留静默未验证回退"的原则,所以按计划推迟。184G 缓存本身在磁盘上
  已不存在(`data/ekfac/cache/...` 为空),所以不存在"占着 184G 不删"的问题,只是代码清理
  顺序问题。

## 没做 / 没跳过

- 未跑 Block C 的任何重计算(等独立 review 清 Block B 再往下,贯彻"先独立验证再往上搭")。
- 全 MLP 的 Delta/温度计未在本 Block 算(Block C 决定单层是否够代表后再定 24 层 Delta 的范围)。

## 独立 sub-agent review（已完成,独立重推真值)

审查者用**与作者完全不同的独立手段**重推了全部五个问题,全部判"对":
1. **多层 Fisher 因子**:它自搭一个不同的两层网,用 `forward_pre_hook`(抓输入)
   + `retain_grad()` 读输出张量 `.grad`(完全绕开被审的 backward hook)独立重建每层
   (m, delta),numpy fp64 累 A/S 真值 —— 与 `accumulate_AS_multi` 逐层 bit-identical
   (0.0e+00),扩到 12 层仍 0.0。
2. **"多层==单层"判据**:独立确认 hook 是纯观察者(挂 0/1/N 个 hook 时 logits 与
   各层 `weight.grad` 逐比特一致);并**独立按 CE 移位定义重推应有的行对齐 label**,
   填补了"两条路径共用同一个错 helper"的共享盲区 —— 行对齐确为对的。
3. **流式块对角影响力代数 + KL-RL 符号**:独立 numpy 重建 `I=-Σ_层 g_f^T F逆 s_m`,
   逐 rollout 等于 `influence_streaming`(rel 2e-15)。
4. **稳健方向**:用余弦定理恒等式独立证 sanity 的数学意义(cos>0 等价于有共同分量),
   并指出它是"必要非充分"(只证减法在做事,残差是否=毒性需下游 IF/探针验证) ——
   作者代码/文档未越界声称。GPU 实跑 24 层印证(primary cos +0.560,‖sub‖/‖tox‖ 0.859)。
5. **缓存free 等价**:独立验证 `inverse_hvp` 对称 PSD(⟨F逆 a,b⟩=⟨a,F逆 b⟩,rel 2e-15),
   据此证明新流式(F逆 搬到 g_f 端)与旧 184G 缓存(g_scaled_z=Q_S^T s_m Q_A/denom)
   **表示同一个量**;现场重算 score 与缓存 s_m 同义、无长度偏置;缺层抛错不静默漏层。

**总评:Block B 机制可信,可作 Block C 基础。**

### review 发现的一处非阻断问题 —— 已修

审查者发现 multilayer / EK-FAC 管线用 `logits.float()` **强制降精度到 fp32** 算 CE,
而 Phase 4 的 `grad_phi` 用 `promote_types(logits.dtype, float32)` **只升不降**(为保 toy
有限差分精度)。**生产 fp32 下是 no-op**(不是 Block C 的正确性 bug),但 fp64 toy 上会
静默把 g_f 精度压到 fp32,是日后 toy 级有限差分验证的隐患。

已按建议把 7 处 score/factor CE 的 `.float()` 统一改成 `promote_types`(factors/eigen/
training/multilayer 一致,与 grad_phi 对齐;pseudo-label 采样的 softmax 不动以免改变离散
抽样)。**复跑全部门确认是 no-op**:Block A 回归门数字逐位不变(0.0130/0.0156/Spearman
0.9983)、多层==单层仍 0.0e+00、流式==直算仍 0.0、audit_1 10/10、toy 套件 13/13 绿。
