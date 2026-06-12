# 度量粒度精确修正:把 Delta 推到与 Fisher 一致的逐 token 粒度

> 这是推导文档,不是代码改动。按用户要求:推导 → 停下给用户看 + 独立 sub-agent
> 验 → 确认后再改代码 → 再重验(Block A toy 门 + gate-2 温度计用新粒度重跑)。
> 数学全部用文字写。记号:F 逆 = inverse-F;逐 token 分数 = 每个 token 位置的
> 分数向量;HVP = Hessian 向量积;优势 A = 同 prompt 内中心化的有效奖励。

## 0. 一句话结论

温度计的分母 F 是**逐 token** 粒度(EK-FAC 标准),但分子 Δ 现在用的是**逐样本
mean-reduction** 粒度,两者差约一个响应长度 R 的尺度——之前靠估一个系数 c 换算
是蒙混。正确做法是把 Δ 也精确重算成逐 token 粒度。推导下来,改动落在三点:
(1) Δ 的"分数外积项"改成**逐 token 位置的对角和**;(2) Δ 的"Hessian 项"改成
**sum-reduction(整序列求和)的 HVP**;(3) Δ 按**总 token 数**归一化。F、g_f、s_m
都不用动。**而且:这个修正可能让温度计显著变小、甚至改变 kill 判据**(见第 5 节,
最重要)。

---

## 1. 问题:三种不同粒度的 Fisher

一个序列的对数似然按定义是**各 token 对数概率之和**:log pi(y|x) = 各 token 的
log pi(y_t | x, y_<t) 求和。所以"真正的逐样本分数"是各 token 分数之和(sum),
不是平均(mean)。但项目代码出于去长度偏置(CLAUDE.md C3)用了 mean。于是有三种
Fisher:

- **逐 token Fisher**(记 F_tok):对所有 token 位置取平均,每个位置一个分数外积。
  **EK-FAC 建的就是它**(`accumulate_AS` 把每个位置的外积累加、再除以总 token 数)。
- **逐样本-mean Fisher**(F_mean):每条序列用 mean 分数(各 token 分数之和除以 R)
  的外积。这是 `compute_per_sample_grad` / 现在的 Δ 用的粒度。
- **逐样本-sum Fisher**(F_sum):每条序列用 sum 分数(各 token 分数之和)的外积。
  这是 IFT 推导里 log pi(y|x) 的分数对应的"数学正确"Fisher。

在"序列内各 token 近似不相关"的近似下(这正是 EK-FAC/K-FAC 的核心近似),它们差
一个标量 R(响应长度):**F_mean : F_tok : F_sum ≈ 1 : R : R的平方**。
(audit 5 实测 F_tok / F_mean ≈ 13–22,正好 ≈ mean R ≈ 22,印证。)

所以现状:F = F_tok(逐 token),Δ 用 mean 粒度的分数和 Hessian。温度计
‖Δ~ p‖/‖g_f‖ 把这两个不同粒度的算子混在一起,绝对值带一个约 R 的尺度,c 只是
事后估这个尺度的近似。

**用户决定(我同意)**:保持 F = F_tok(EK-FAC/Grosse/MDA 逐 token 标准,毕业论文
约定),把 Δ 精确重算到逐 token 粒度去对齐它。下面是推导。

---

## 2. Delta 是什么(复述,确认起点)

IFT 推导给出 dG/dθ = Δ − βF,其中(s 是分数向量,H 是 log pi 的 Hessian):
- F = 期望 of (s s 外积),
- **Δ = 期望_{x, y采样自 pi_θ*} of [ A(x,y) · ( s s外积 + H ) ]**,
  A(x,y) = 该序列的有效奖励减同 prompt 均值(逐序列的标量优势)。

关键性质(后面要用):**Fisher 恒等式** 在序列层级成立——
期望_{y采样自 pi} of ( s_sum s_sum外积 + H_sum ) = 0,即"分数外积的期望 = 负的
Hessian 期望"。所以当 A2 成立(A 对所有 y 恒为 0)时 Δ = 0;A2 失效时,Δ 只拾起
"曲率与优势 A 的协变部分"(因为 A 零均值,常数曲率乘 A 期望为 0)。

注意 Δ 里的 s、H 按 log pi(y|x) 的定义都是**整序列**的:s = 各 token 分数之和
s_sum,H = 各 token Hessian 之和 H_sum。

---

## 3. 推导:逐 token 粒度的 Delta

目标:把 Δ 写成和 F_tok 同构的"对所有 token 位置取平均"的算子。

### 3.1 把序列级的 Δ 拆到 token 级

F_tok = (1/总token数) · 对所有 token 位置 u 求和 of (g_u g_u外积),
其中 **g_u = outer(delta_u, m_u)** 是位置 u 的逐 token 梯度:m_u 是该层在位置 u 的
输入,delta_u 是 **sum-reduction 反传**下该层在位置 u 输出处的梯度。
(这正是 `accumulate_AS` 的做法——它用 reduction="sum" 反传、按位置取 outer、
除以总 token 数。)

与之同构的逐 token Δ:
> **Δ_tok = (1/总token数) · 对所有 token 位置 u 求和 of [ A(seq(u)) · ( g_u g_u外积 + h_u ) ]**

其中每个 token 位置 u 携带它所属序列的优势 A(seq(u)),g_u 同上,h_u 是位置 u 的
逐 token Hessian。Δ_tok 和 F_tok 都是"对相同 token 位置集合取平均",同一个 g_u
表示、同一个除以总 token 数的归一化——**这就是粒度一致**。

### 3.2 两项的具体算法(matrix-free,对任意向量 v)

Δ_tok 作用在向量 v 上,按序列分组(A 在序列内为常数,可提出来):
> **Δ_tok · v = (1/总token数) · 对每条序列求和 of A(seq) · [ 对角分数项(v) + 整序列HVP(v) ]**

**(a) 对角分数项**:对该序列响应窗口内每个位置 u 求和 of g_u · (g_u 点乘 v)。
其中 g_u 点乘 v = delta_u 点乘 (v 乘 m_u)(把 v 当 (d_out, d_in) 矩阵),是标量;
g_u · 标量 = outer(delta_u, m_u) × 标量。这一项**只用一次反传里 hook 抓到的逐
token (m_u, delta_u)**,不需要额外反传。
- **为什么是"对角和"而不是"整序列分数的外积"**:整序列 sum 分数 s_sum = 各 g_u
  之和,s_sum(s_sum 点乘 v) 会含跨 token 交叉项(u≠u')。而 F_tok 恰恰**丢掉**了
  这些跨 token 交叉项(只保对角 u=u')。为了和 F_tok 一致,Δ 的分数项也必须只取
  对角和。**这是和 F_tok 完全相同的"序列内 token 不相关"近似**,所以两个算子
  一致。(这是本推导里唯一的近似选择,第 4 节论证它为何是正确的一致性选择。)

**(b) 整序列 HVP 项**:HVP of (响应 token 的 log pi 之和) 作用于 v,即
grad^2_φ ( 各响应 token 的 log pi 求和 ) · v。一次二阶反传(double-backward,
**reduction="sum"**)。
- **为什么 H 项不需要丢交叉项**:Hessian of (和) = (各 Hessian) 之和,**精确无
  近似**、本来就没有跨 token 交叉项。所以整序列 sum 的 HVP 精确等于各 token
  Hessian h_u 之和,直接就是 Δ_tok 要的 sum_u h_u。

把 (a)+(b) 乘以该序列的 A、对所有序列求和、除以总 token 数,就是 Δ_tok · v。

### 3.3 和现状代码的差别(只动 Δ 端)

现在的 `phase4/hvp_logpi.py` + `e3_delta.py`:
- 分数项用 **整序列 mean 分数** s_mean 的外积 s_mean(s_mean 点乘 v) —— 错两处:
  (i) 是 mean 不是逐 token 对角;(ii) 含交叉项且被 1/R² 压小。
- Hessian 项用 **mean-reduction** 的 HVP —— 比正确的 sum HVP 小 1/R。
- 归一化是逐 prompt / 逐样本平均 —— 不是逐 token。

改成:分数项 = 逐 token 对角和(hook 抓 m_u/delta_u);HVP = **sum-reduction**;
归一化除以总 token 数。**F(accumulate_AS)、g_f、s_m 都不动**。

---

## 4. 为什么这是"精确一致",以及为什么不再需要 c

1. **同粒度**:Δ_tok 和 F_tok 都是"对相同 token 位置集合、用相同 g_u 表示、除以
   相同总 token 数"的平均算子。温度计 ‖Δ_tok~ p‖/‖F_tok p‖ 是同粒度算子之比,
   那个约 R 的尺度**从构造上消掉了**,不需要事后乘 c。这就是用户要的"分子分母
   同一种粒度精确重算"。
2. **A2 抵消自动成立**:Δ_tok 正比于 A,A2 成立时 A 恒为 0 → Δ_tok = 0,与 g_u
   的具体表示无关。
3. **唯一的近似(丢跨 token 交叉项)和 F_tok 的近似是同一个**:F_tok 从 F_sum
   丢掉的、与 Δ 分数项丢掉的,是**同一批**跨 token 交叉项。既然我们选定用 F_tok
   当曲率模型(EK-FAC 标准),对 Δ 施加同样的近似才自洽。H 项无此问题(精确)。
4. **温度计对 g_f / s_m 的缩放不变**:温度计 = ‖Δ~ p‖/‖g_f‖,p = inverse-F · g_f。
   把 g_f 乘任意常数 α,p、Δ~ p、分母同步乘 α,比值不变。所以 g_f / s_m 用 mean
   还是 sum 不影响温度计——**唯一要对齐的就是 Δ 和 F 这两个算子**(现已对齐)。
   生产影响力**排序**同样对全局缩放不变,不受影响。

---

## 5. 【最重要】这个修正可能改变 kill 判据,必须重测、不带预期

推导中发现一件比"差 R 因子"严重得多的事:

> 【独立验证者修正本节】初稿这里写过一个**方向性预测**("修正后 Δ 很可能显著
> 变小、kill 可能翻 transition"),独立 sub-agent 用非线性玩具给了反例、证明该
> 预测不可靠,已删除。下面是修正后的中性表述。

现在的 mean-reduction 让 Δ 的**分数项被压小约 R 倍**(分数项带 1/R²、Hessian 项
带 1/R,差 R)。这有两层后果:

1. **Δ 与 F 不同粒度**(分数项逐样本-mean,F 逐 token)——温度计绝对值带约 R 的
   尺度。这是确凿的错误。
2. **Δ 内部两项失衡**:Fisher 恒等式说"分数外积的期望 = 负 Hessian 的期望"(序列
   层级与 token 层级都成立,验证者机器精度确认)。在**正确的同尺度(逐 token)**下,
   分数项与 Hessian 项**量级相当**(玩具实测两项范数 0.3128 vs 0.3123),会发生
   **部分抵消**(实测抵消率约 0.81,即远非完全抵消)。mean-reduction 把分数项压小
   R 倍后,变成 **Hessian 项主导**、抵消更少。

修正成逐 token(sum)粒度后,分数项回到与 Hessian 同尺度,两项的部分抵消按其正确
结构恢复。**但"抵消恢复"不蕴含"Δ 变小"**:两项同尺度后可能相消(Δ 变小)、也可能
同号叠加(Δ 变大)。验证者在非线性玩具上实测到 **Δ 反而变大**(mean 范数 0.346 →
tok 范数 0.509)的反例——所以净效果是**方向不定**的,取决于"曲率与优势 A 的协变
结构"这个经验量。

**结论(中性,这是唯一诚实的判断)**:
- 确凿:现状 mean 粒度的 Δ 在数学上**错**(与 F 不同粒度、内部两项差 R 倍);逐
  token 形式是与 F_tok 一致的正确形式。
- **不预测**修正后温度计变大还是变小、kill 还是 transition——只能实测。
- 因此 **Phase 4 的 kill / OLMo 的 DEATH 在新粒度下可能改变(任一方向)**,必须用
  新 Δ 重跑 gate-2,**数字说什么报什么**。若结论变了,那是修对了之后的真结论,
  不是降标准;若不变,kill 更稳。这正是用户说"OLMo 迟早撞死线、必须现在把代码
  改对"的原因——边界情况下,粒度对不对直接决定过没过 0.3。

---

## 6. 改什么(确认后再动手)

- `phase4/hvp_logpi.py`:`_mean_logpi` → sum reduction;`score_and_hvps` 增加 hook
  抓逐 token (m_u, delta_u)(在响应窗口),返回逐 token 对角分数项所需量 + sum-HVP。
- `phase4/e3_delta.py`:`_contrib` 把"s(s^T v)"换成"逐 token 对角和";per-prompt
  归一化改成除以总 token 数以对齐 F_tok。
- `run_gptneo_gate2_killcheck.py`:去掉 c 估计和"度量自洽 ≈ c×raw"那栏,直接报
  同粒度温度计。
- **不动**:`ekfac/`(F_tok、g_f、s_m 全不变)。

## 7. 验证计划(改完之后)

1. **toy 上 brute-force 逐 token Δ**:小模型上显式按定义 Δ_tok = (1/总token数)·
   Σ_u A·(g_u g_u外积 + h_u) 造出稠密 Δ_tok 矩阵,验证 matrix-free Δ_tok·v 与它
   逐元素一致(验证者已预验,误差 3.3e-16);并验证"同粒度"性质(F_tok 与 Δ_tok 在
   同一 token 归一化下,把序列长度 R 人为放大不改变温度计——即 c→1)。
   **追加体检指标(验证者建议)**:报告 Δ 的分数项范数与 Hessian 项范数——在逐
   token 粒度下两者应**同量级**(玩具上 0.31 vs 0.31);若分数项远小于 Hessian 项,
   说明又退回了 mean-粒度的失衡,是回归哨兵。
2. **Block A toy 门**:不受影响(Δ 不在 Block A 里),复跑确认仍绿。
3. **独立 sub-agent 重验**:从 IFT 定义独立重推逐 token Δ 形式,核对(a) 对角分数
   项、(b) sum-HVP、(c) 总 token 归一化、(d) 跨 token 近似与 F_tok 一致这四点。
4. **gate-2 温度计用新 Δ 重跑**:报同粒度温度计;诚实判读 kill 是否仍成立。
