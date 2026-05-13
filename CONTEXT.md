# KL-Regularized RLHF-PPO Influence Function：从单个 Prompt 到工程实验的推进方案

## 0. 目标

本文档的目标是把 KL-regularized RLHF-PPO 场景下的 influence function 推导和工程实现路线整理成一个可逐步推进的研究方案。

当前我们先不处理完整的 $\mathbb E_{x\sim\mathcal D}$，而是先固定一个 prompt $x$，把单个 prompt 下的 rollout-level influence score 算清楚。然后再把它推广到整个 prompt 数据分布。

核心问题是：

> 在 PPO/RLHF 训练后，一个训练 rollout $(x, y_{1:T})$ 对某个 eval 样本的 log probability 有多大影响？

我们希望得到一个可计算的 score：

$$
\mathcal I_f(x,y_m)
=
-
\nabla_\phi f(\theta^*)^\top
F_{x,\phi}^{-1}
\nabla_\phi \log \pi_{\theta^*}(y_m|x)
$$

其中：

- $x$：固定的训练 prompt；
- $y_m=(y_{m,1},\dots,y_{m,T})$：PPO rollout 出来的完整 token 序列；
- $f$：我们关心的 eval 目标，例如 eval 样本的 log probability；
- $\phi$：我们选择分析的参数子空间，例如 最后几层 MLP、某些 probe-relevant layers；
- $F_{x,\phi}$：在固定 prompt $x$ 下，对 rollout 分布估计出来的 Fisher；
- $\theta^*$：PPO/RLHF 训练后的模型参数。

---

## 1. 固定单个 Prompt 的 KL-Regularized RL 目标

先固定一个 prompt $x$。

PPO/RLHF 的理想化 KL-regularized objective 可以写成：

$$
J_x(\theta)
=
\mathbb E_{y\sim\pi_\theta(\cdot|x)}
\left[
r(x,y)
-
\beta
\log
\frac{\pi_\theta(y|x)}{\pi_{\mathrm{ref}}(y|x)}
\right]
$$

这里 $y$ 不是单个 token，而是一整条 response sequence：

$$
y=(y_1,\dots,y_T)
$$

因此：

$$
\log \pi_\theta(y|x)
=
\sum_{t=1}^{T}
\log \pi_\theta(y_t|x,y_{<t})
$$

同样：

$$
\log \pi_{\mathrm{ref}}(y|x)
=
\sum_{t=1}^{T}
\log \pi_{\mathrm{ref}}(y_t|x,y_{<t})
$$

定义 effective reward：

$$
\tilde R(x,y,\theta)
=
r(x,y)
-
\beta
\log
\frac{\pi_\theta(y|x)}{\pi_{\mathrm{ref}}(y|x)}
$$

展开到 token level：

$$
\tilde R(x,y,\theta)
=
r(x,y)
-
\beta
\sum_{t=1}^{T}
\left[
\log \pi_\theta(y_t|x,y_{<t})
-
\log \pi_{\mathrm{ref}}(y_t|x,y_{<t})
\right]
$$

注意：这里的 $r(x,y)$ 通常是 reward model 给整条 response 的 sequence-level reward。

> **关于 KL 项的工程估计**：上式中的 $\log\pi_\theta(y_t|\cdot) - \log\pi_\text{ref}(y_t|\cdot)$ 是真 KL 散度在单 token 上的 **k1 估计器**（Schulman 2020）。k1 数学上无偏（$\mathbb{E}_{y\sim\pi_\theta}[\log\pi_\theta/\pi_\text{ref}] = D_\text{KL}$），但单次 sample 可正可负。在 detoxification 场景下，policy 系统性压制 toxic token 到极低概率，导致 k1 在被压制 token 上系统性为大负值；TRL 的 PPO 实现中 `non_score_reward = -β·k1` 在这些 token 上从"惩罚"翻号成"奖励"，训练失稳。
>
> 我们在工程实现中**统一采用 k2 估计器**：
>
> $$\hat D_\text{KL,k2}(y_t|x,y_{<t}) = \frac{1}{2}\left[\log\pi_\theta(y_t|x,y_{<t}) - \log\pi_\text{ref}(y_t|x,y_{<t})\right]^2$$
>
> k2 恒非负、方差低，作为 reward shaping 项行为正确。k2 是有偏估计（Jensen 不等式下高估真 KL），因此 PPO 收敛到的 policy 不是严格意义上的 Section 2 中 KL-RL 最优解 $\pi^*$，而是对应**有效 β' > β** 的 KL-RL 最优点。但 Section 5/7 的推导只依赖 "policy 是 KL-RL 最优点" 这一结构性假设，不依赖 β 的精确数值，因此 IF 公式和相消假设在 k2 训练下完全成立——**前提是 Phase 1 effective reward variance 诊断时也使用 k2 来计算 effective reward**（见 Section 8）。
>
> 后续公式中出现的 $\log\pi_\theta/\pi_\text{ref}$ 默认指其数学定义；工程实现处会显式指明使用 k2 形式。

---

## 2. 单 Prompt 下的最优策略

固定 $x$ 后，KL-regularized RL 的闭式最优策略是：

$$
\pi_x^*(y)
=
\frac{
\pi_{\mathrm{ref}}(y|x)
\exp(r(x,y)/\beta)
}{
Z(x)
}
$$

其中：

$$
Z(x)
=
\sum_y
\pi_{\mathrm{ref}}(y|x)
\exp(r(x,y)/\beta)
$$

等价地，反解 reward：

$$
r(x,y)
=
\beta
\log
\frac{\pi_x^*(y)}{\pi_{\mathrm{ref}}(y|x)}
+
\beta \log Z(x)
$$

代入 effective reward：

$$
\tilde R(x,y,\theta^*)
=
r(x,y)
-
\beta
\log
\frac{\pi_x^*(y)}{\pi_{\mathrm{ref}}(y|x)}
=
\beta\log Z(x)
$$

因此，在理想最优点处：

$$
\tilde R(x,y,\theta^*)
$$

只依赖 $x$，不依赖 $y$。

这一步是整个理论的核心：

> 对同一个 prompt $x$，如果 policy 已经接近 KL-RL 最优点，那么不同 rollout $y$ 的 effective reward 应该近似相等。

---

## 3. 单个 Prompt 下的 Score Function

对一条 rollout：

$$
y_m=(y_{m,1},\dots,y_{m,T})
$$

它的 sequence-level log probability 是：

$$
\log \pi_\theta(y_m|x)
=
\sum_{t=1}^{T}
\log \pi_\theta(y_{m,t}|x,y_{m,<t})
$$

对应的 score function（数学定义）是：

$$
s_m
=
\nabla_\theta \log \pi_\theta(y_m|x)
=
\sum_{t=1}^{T}
\nabla_\theta
\log \pi_\theta(y_{m,t}|x,y_{m,<t})
$$

> **工程注记（长度归一化）**：原始定义 $s_m = \sum_{t=1}^{T} \nabla_\phi \log\pi_\theta(y_{m,t}|\cdot)$ 与序列长度 $T$ 正相关，可能导致较长的 rollout 在 influence ranking 中系统性偏高。建议在实现时改用 per-token 平均：
>
> $$s_{m,\phi} = \frac{1}{T} \sum_{t=1}^{T} \nabla_\phi \log\pi_\theta(y_{m,t}|x, y_{m,<t})$$
>
> eval 梯度同理：
>
> $$g_{\text{eval},\phi} = \frac{1}{L} \sum_{t=1}^{L} \nabla_\phi \log\pi_\theta(y_{\text{eval},t}|x_{\text{eval}}, y_{\text{eval},<t})$$
>
> 两侧同时做归一化后，influence score 的含义变为"每个 token 平均的相互影响"，跨 rollout 比较时更公平。实现上只需在 backward 之前把 loss 的 reduction 改为 `mean`（而非 `sum`）即可，不需要改变其他任何计算流程。详见 Section 18.6。

如果我们只分析某个参数子空间 $\phi$，例如 某些 MLP 层，则写成：

$$
s_{m,\phi}
=
\nabla_\phi \log \pi_\theta(y_m|x)
$$

---

## 4. 单 Prompt Fisher 的定义

在固定 prompt $x$ 下，Fisher 定义为：

$$
F_x(\theta^*)
=
\mathbb E_{y\sim\pi_{\theta^*}(\cdot|x)}
\left[
\nabla_\theta\log\pi_{\theta^*}(y|x)
\nabla_\theta\log\pi_{\theta^*}(y|x)^\top
\right]
$$

如果只看参数子空间 $\phi$：

$$
F_{x,\phi}(\theta^*)
=
\mathbb E_{y\sim\pi_{\theta^*}(\cdot|x)}
\left[
\nabla_\phi\log\pi_{\theta^*}(y|x)
\nabla_\phi\log\pi_{\theta^*}(y|x)^\top
\right]
$$

在工程上，$F_x$ 可以通过对同一个 prompt $x$ 采样多条 rollout 来估计：

$$
\hat F_{x,\phi}
=
\frac{1}{K}
\sum_{k=1}^{K}
 s_{k,\phi}s_{k,\phi}^\top
$$

其中：

$$
s_{k,\phi}
=
\nabla_\phi\log\pi_{\theta^*}(y^{(k)}|x)
$$

---

## 5. 单 Prompt 下的 Influence Function

现在考虑对某条训练 rollout $z_m=(x,y_m)$ 加一个小权重 $\epsilon$。扰动后的目标为：

$$
J_{x,\epsilon}(\theta)
=
J_x(\theta)
+
\epsilon
\left[
r(x,y_m)
-
\beta
\log
\frac{\pi_\theta(y_m|x)}{\pi_{\mathrm{ref}}(y_m|x)}
\right]
$$

在 KL-RL 最优点近似成立的条件下，有：

$$
\frac{d\theta^*}{d\epsilon}
=
-
F_x(\theta^*)^{-1}
\nabla_\theta\log\pi_{\theta^*}(y_m|x)
$$

限制到参数子空间 $\phi$：

$$
\frac{d\phi^*}{d\epsilon}
=
-
F_{x,\phi}(\theta^*)^{-1}
\nabla_\phi\log\pi_{\theta^*}(y_m|x)
$$

因此，对任意目标函数 $f(\theta)$，rollout $(x,y_m)$ 的 influence score 是：

$$
\boxed{
\mathcal I_f^x(x,y_m)
=
-
\nabla_\phi f(\theta^*)^\top
F_{x,\phi}(\theta^*)^{-1}
\nabla_\phi\log\pi_{\theta^*}(y_m|x)
}
$$

这就是单 prompt 下的 PPO/RLHF rollout-level influence score。

---

## 6. 当目标 $f$ 是 Eval 样本的 Log Probability

现在选择一个具体 eval 样本：

$$
z_{\mathrm{eval}}
=(x_{\mathrm{eval}},y_{\mathrm{eval}})
$$

其中：

$$
y_{\mathrm{eval}}=(y_1,\dots,y_L)
$$

定义目标函数：

$$
f(\theta)
=
\log \pi_\theta(y_{\mathrm{eval}}|x_{\mathrm{eval}})
$$

展开为：

$$
f(\theta)
=
\sum_{t=1}^{L}
\log
\pi_\theta(y_t|x_{\mathrm{eval}},y_{eval,<t})
$$

它的梯度是 eval sequence 的 score：

$$
g_{\mathrm{eval},\phi}
=
\nabla_\phi f(\theta^*)
=
\nabla_\phi
\log \pi_{\theta^*}(y_{\mathrm{eval}}|x_{\mathrm{eval}})
$$

因此，单 prompt 下训练 rollout $(x,y_m)$ 对 eval logprob 的 influence 是：

$$
\boxed{
\mathcal I_{\mathrm{eval}}^x(x,y_m)
=
-
g_{\mathrm{eval},\phi}^\top
F_{x,\phi}^{-1}
s_{m,\phi}
}
$$

其中：

$$
s_{m,\phi}
=
\nabla_\phi
\log \pi_{\theta^*}(y_m|x)
$$

解释：

- 如果 $\mathcal I_{\mathrm{eval}}^x(x,y_m)>0$，增加这个 rollout 的权重会提高 eval sequence 的 log probability；
- 如果 $\mathcal I_{\mathrm{eval}}^x(x,y_m)<0$，增加这个 rollout 的权重会降低 eval sequence 的 log probability。

注意：如果目标改成 eval negative log likelihood：

$$
f(\theta)=-\log \pi_\theta(y_{\mathrm{eval}}|x_{\mathrm{eval}})
$$

则符号解释会反过来。

---

## 7. Effective Reward Variance 是什么？(先把前面的实验跑起来最重要, 验证理论合理性那是后面的事情)

我们希望检查当前 PPO/RLHF 训练后的 policy 是否接近 KL-regularized RL 的理想最优点。

理论上，在固定 prompt $x$ 下，如果 $\pi_\theta\approx\pi_x^*$，则：

$$
\tilde R(x,y,\theta)
\approx
\beta\log Z(x)
$$

也就是说，对同一个 $x$，不同 rollout $y$ 的 effective reward 应该差不多。

因此我们可以计算：

$$
\operatorname{Var}_{y\sim\pi_\theta(\cdot|x)}
\left[
\tilde R(x,y,\theta)
\right]
$$

这个量的含义是：

> 固定一个 prompt $x$，从 PPO 后模型中采样多条 response，然后看这些 response 的 effective reward 在不同 response 之间波动有多大。

---

## 8. Effective Reward Variance 的实际计算步骤

对一个 prompt $x$：

### Step 1：采样多条 rollout

从训练后的 policy $\pi_\theta$ 中采样：

$$
y^{(1)},\dots,y^{(K)}
$$

### Step 2：对每条 rollout 计算 reward

使用 reward model 得到：

$$
r(x,y^{(k)})
$$

### Step 3：计算 policy logprob

$$
\log \pi_\theta(y^{(k)}|x)
=
\sum_t
\log \pi_\theta(y_t^{(k)}|x,y_{<t}^{(k)})
$$

### Step 4：计算 reference logprob

$$
\log \pi_{\mathrm{ref}}(y^{(k)}|x)
=
\sum_t
\log \pi_{\mathrm{ref}}(y_t^{(k)}|x,y_{<t}^{(k)})
$$

### Step 5：计算 effective reward（k2 估计器）

按 Section 1 的工程约定，使用 k2 估计器：

$$
\tilde R^{(k)}
=
r(x,y^{(k)})
-
\beta
\cdot
\frac{1}{2}
\sum_{t=1}^{T_k}
\left[
\log \pi_\theta(y^{(k)}_t|x, y^{(k)}_{<t})
-
\log \pi_{\mathrm{ref}}(y^{(k)}_t|x, y^{(k)}_{<t})
\right]^2
$$

注意：sum 是在 response token 上，与训练 reward shaping 一致。**Phase 0 训练已计划使用 k2 reward shaping，此处必须用同样的 k2 公式**——否则训练目标和诊断目标不一致，相消假设失效。

### Step 6：计算方差

$$
\widehat{\operatorname{Var}}_x
=
\frac{1}{K-1}
\sum_{k=1}^{K}
\left(
\tilde R^{(k)}
-
\overline{\tilde R}_x
\right)^2
$$

其中：

$$
\overline{\tilde R}_x
=
\frac{1}{K}
\sum_{k=1}^{K}
\tilde R^{(k)}
$$

---

## 9. Effective Reward Variance 的解释

如果 $\widehat{\operatorname{Var}}_x$ 很小，说明：

> 对这个 prompt $x$，不同 rollout 的 effective reward 近似相等。

这支持下面的理论近似：

$$
\tilde R(x,y,\theta^*)
\approx
\beta\log Z(x)
$$

也就是说，当前 policy 在这个 prompt 附近比较接近 KL-RL 最优解。

如果 $\widehat{\operatorname{Var}}_x$ 很大，说明：

> 当前模型对这个 prompt 还没有达到理论推导所依赖的局部最优结构。

这时：

$$
\frac{\partial G}{\partial \theta}\bigg|_{\theta^*}
\approx
-\beta F
$$

这个近似可能不好，最终 influence score 也可能不准。

重要注意：

> 方差应该在同一个 prompt $x$ 内部，对不同 rollout $y$ 计算。不要把不同 prompt 的 rollout 混在一起直接算一个总 variance。

因为不同 prompt 的 $\beta\log Z(x)$ 本来就可以不同。

---

## 10. 从单 Prompt 到多个 Prompt

单 prompt 版本使用：

$$
F_{x,\phi}
=
\mathbb E_{y\sim\pi_\theta(\cdot|x)}
[s_\phi(x,y)s_\phi(x,y)^\top]
$$

完整数据集版本应该使用：

$$
F_{\mathcal D,\phi}
=
\mathbb E_{x\sim\mathcal D,\,y\sim\pi_\theta(\cdot|x)}
[s_\phi(x,y)s_\phi(x,y)^\top]
$$

其中：

$$
s_\phi(x,y)
=
\nabla_\phi \log\pi_\theta(y|x)
$$

于是全局 influence 写成：

$$
\boxed{
\mathcal I_{\mathrm{eval}}(x_m,y_m)
=
-
g_{\mathrm{eval},\phi}^\top
F_{\mathcal D,\phi}^{-1}
s_{m,\phi}
}
$$

这更适合真实工程实验，因为全局 Fisher 比单 prompt Fisher 稳定。

建议：

- 理论验证阶段：使用 $F_{x,\phi}$；
- 工程实验阶段：使用 $F_{\mathcal D,\phi}$。

---

## 11. 参数子空间 $\phi$：为什么不需要全模型？

全模型 Fisher 维度极高，几乎不可直接计算。

因此只选择某些参数子空间。本项目固定使用 **Layer 19 的 MLP $W_2$**（理由见 Section 12）。其他可选范围（仅作背景，不在本项目执行范围内）：最后一层 MLP、与 toxic vector 高相关的其他层、PPO 实际更新较大的层、LoRA adapter 参数等。

此时 influence 不再是全模型 influence，而是：

> 在选定参数子空间 $\phi$ 中，某条 PPO rollout 对 eval logprob 的局部影响。

这在工程上更可行，也更容易和 mechanistic interpretability 的 probe 分析结合。

> **工程约束**：EK-FAC 的 Kronecker 分解严格依赖线性层结构（$\text{out} = Wx$），因此 $\phi$ 只能包含线性变换参数（MLP 的 $W_1$/$W_2$、attention 的 QKV/O projection）。LayerNorm 参数和 embedding 参数没有对应的 Kronecker 结构，若要包含需自行手写 Fisher 近似。MLP 的 $W_2$（value matrix）是标准线性层，完全兼容 EK-FAC。参考 MDA（Chen et al., 2026）的实现，其 `ekfac_blocks.py` 对 attention 的 QK/QKVO 矩阵做因子拟合；扩展到 MLP $W_2$ 结构完全平行。

---

## 12. 如果只看某层 MLP 的 $W_2$（Value Matrix），Score 怎么算？

> **符号说明**：本节沿用 Lee et al. (2024) 的记法，用 $W_1$ 表示 MLP 的第一个线性层（对应其论文中的 $W_K$，key matrix），$W_2$ 表示第二个线性层（对应其论文中的 $W_V$，value matrix）。注意这与标准 Transformer 文献中 $W_K$/$W_V$ 表示 attention 投影矩阵的用法不同，请勿混淆。

我们选择某一层 $\ell$（例如论文中标注为 toxic 的 Layer 19）的 MLP value matrix $W_2^\ell$ 作为参数子空间 $\phi$。

---

### MLP 结构回顾

第 $\ell$ 层 MLP 的计算为：

$$
\text{out}_t^\ell = W_2^\ell \cdot m_t^\ell
$$

其中：

$$
m_t^\ell = \sigma\!\left(W_1^\ell \, h_t^{\ell\text{-mid}}\right) \in \mathbb{R}^{d_\text{mlp}}
$$

- $h_t^{\ell\text{-mid}} \in \mathbb{R}^d$：第 $\ell$ 层 attention 之后、MLP 之前的 residual stream（位置 $t$）；
- $m_t^\ell$：经过 $W_1$ 线性变换再经激活函数后的 **post-activation 向量**（即 MLP 隐层激活）；
- $W_2^\ell \in \mathbb{R}^{d \times d_\text{mlp}}$：value matrix，每一列 $v_i^\ell$ 是一个 value vector；
- $\text{out}_t^\ell \in \mathbb{R}^d$：MLP 输出，加回 residual stream。

---

### $W_2$ 的梯度公式

对单个 token 位置 $t$，定义 **下游梯度**：

$$
\delta_t^\ell
= \frac{\partial \log \pi_\theta(y_t \mid x, y_{<t})}{\partial \,\text{out}_t^\ell}
\in \mathbb{R}^d
$$

即 $\log \pi$ 通过 MLP $\ell$ 输出端反向传播回来的梯度向量。

则 $W_2^\ell$ 的梯度为：

$$
\nabla_{W_2^\ell} \log \pi_\theta(y_t \mid x, y_{<t})
=
\delta_t^\ell \cdot (m_t^\ell)^\top
\in \mathbb{R}^{d \times d_\text{mlp}}
$$

对整条 sequence $y = (y_1, \dots, y_T)$，score 矩阵（数学定义，sum 形式）为：

$$
s_{W_2}(x, y)
=
\sum_{t=1}^{T}
\delta_t^\ell \cdot (m_t^\ell)^\top
\in \mathbb{R}^{d \times d_\text{mlp}}
$$

> **工程实现**：实际计算时改为 per-token 平均（见 Section 3 工程注记及 Section 18.6）：
> $$s_{W_2}(x, y) = \frac{1}{T}\sum_{t=1}^{T} \delta_t^\ell \cdot (m_t^\ell)^\top$$

---

### 两个量的来源

**$m_t^\ell$（上游激活，forward pass 得到）**：

MLP 隐层激活，即 $W_1$ 线性变换后经激活函数的结果。可通过在 MLP 隐层注册 forward hook 获取：

```python
mlp_activations = {}

def hook_fn(module, input, output):
    mlp_activations[layer_idx] = output  # shape: (batch, seq_len, d_mlp)

model.transformer.h[layer_idx].mlp.act.register_forward_hook(hook_fn)
```

**$\delta_t^\ell$（下游梯度，backward pass 得到）**：

$\log \pi(y|x)$ 对 MLP $\ell$ 输出的梯度，需在 MLP 输出端注册 backward hook：

```python
mlp_grad_outputs = {}

def grad_hook_fn(module, grad_input, grad_output):
    mlp_grad_outputs[layer_idx] = grad_output[0]  # shape: (batch, seq_len, d)

model.transformer.h[layer_idx].mlp.register_full_backward_hook(grad_hook_fn)
```

---

### Score 的计算

```python
# m_t: (seq_len, d_mlp)
# delta_t: (seq_len, d)

# per-token 平均（见 Section 18.6）
s_W2 = torch.einsum('td,tm->dm', delta_t, m_t) / seq_len  # shape: (d, d_mlp)

# flatten 成向量，用于后续 influence 计算
s_W2_flat = s_W2.reshape(-1)  # shape: (d * d_mlp,)
```

对训练 rollout $(x, y_m)$：

$$
s_{m, W_2}
=
\frac{1}{T}\sum_{t=1}^{T}
\delta_{m,t}^\ell \cdot (m_{m,t}^\ell)^\top
$$

对 eval 样本 $(x_\text{eval}, y_\text{eval})$：

$$
g_{\text{eval}, W_2}
=
\frac{1}{L}\sum_{t=1}^{L}
\delta_{\text{eval},t}^\ell \cdot (m_{\text{eval},t}^\ell)^\top
$$

---

### Influence Score

$$
\mathcal{I}_\text{eval}(x, y_m)
=
-\,
g_{\text{eval}, W_2}^\top
\,F_{W_2}^{-1}\,
s_{m, W_2}
$$

其中内积理解为把两个 $(d \times d_\text{mlp})$ 矩阵 flatten 后做向量内积。

---

### 总结

|  | MLP $W_2$（本节）|
|---|---|
| 参数 $\phi$ | $W_2^\ell \in \mathbb{R}^{d \times d_\text{mlp}}$ |
| 上游激活 | $m_t^\ell = \sigma(W_1 h_t^{\ell\text{-mid}})$ |
| 下游梯度 | $\delta_t^\ell$（需 backward hook）|
| Score 结构 | $\frac{1}{T}\sum_t \delta_t^\ell (m_t^\ell)^\top$ |
| 获取方式 | 需要完整 backward pass |

---

### 注意事项

1. **层的选择**：$\ell$ 应选择 mechanistic interpretability 分析中标注的 toxic 层（如 Layer 19、Layer 12 等），可基于与 toxic probe $W_\text{toxic}$ cosine similarity 最高的 value vectors 所在层来确定。

2. **多层扩展**：若需覆盖多个 toxic 层，可对每层分别计算 $s_{W_2^\ell}$，然后 concatenate 成一个整体向量，Fisher 对应扩展到各层参数的 block-diagonal 近似。

3. **$\delta_t^\ell$ 的含义**：它是 $\log\pi_\theta(y_t|x,y_{<t})$ 关于 MLP $\ell$ 输出的反向梯度，包含从 LM head 到第 $\ell$ 层之间所有 MLP 和 attention 层的影响，是一个综合了多层信号的中间量。

4. **Prompt Mask（RL setting 独有）**：每条数据结构为 `[prompt tokens] + [response tokens]`。计算 $m_t^\ell$ 和 $\delta_t^\ell$ 时，只对 response 部分的 token 位置做梯度计算（prompt 部分的 label 设为 `-100`，使用 `ignore_index=-100`）。漏掉这个 mask 会导致 Fisher 估计混入 prompt 的曲率信息，score 失去对 rollout 质量的分辨力。详见 Section 14 差异一。

---

## 13. Fisher 矩阵的近似方法

真实的 $F^{-1}$ 在 GPT2-medium 这种规模上无法直接计算。本项目使用以下两个方法：

### 13.1 Gradient Similarity（Sanity Baseline）

最简单的对照：

$$
\mathcal I_{\mathrm{grad}}(m)
=
-
g_{\mathrm{eval}}^\top s_m
$$

不使用任何 Fisher inverse，直接做点积。**作用是诊断 EK-FAC 实现是否真的有效**：

- 如果 EK-FAC 的 ranking 和 gradient similarity 完全相同，说明 Fisher inverse 没有起任何作用（damping 过大、特征分解出错、或实现 bug）；
- 如果 EK-FAC 的 ranking 和 gradient similarity 完全不相关，说明 Fisher 估计错了，方差噪声压过了信号；
- 健康状态是两者**相关但不等同**——EK-FAC 应该改变 ranking 的具体顺序但不会颠倒整体方向。

实现成本几乎为零：在 EK-FAC pipeline 中，$g_\text{eval}$ 和 $s_m$ 都已经计算出来，多做一个点积即可。

### 13.2 EK-FAC（主方法）

EK-FAC（Eigenvalue-corrected Kronecker-Factored Approximate Curvature）是当前 LLM influence function 工程实现中精度最高的 Fisher inverse 近似方法（Grosse et al., 2023；MDA, Chen et al., 2026）。

#### 针对 MLP $W_2^\ell$ 的具体公式

对第 $\ell$ 层 MLP 的 value matrix $W_2^\ell \in \mathbb{R}^{d \times d_\text{mlp}}$，其线性变换为：

$$\text{out} = W_2^\ell \cdot m^\ell, \quad m^\ell = \sigma(W_1^\ell h) \in \mathbb{R}^{d_\text{mlp}}$$

其中 $m^\ell$ 是输入激活（post-activation），$\delta = \frac{\partial \mathcal{L}}{\partial \text{out}} \in \mathbb{R}^{d}$ 是输出处的反向梯度。

#### Kronecker 因子分解

EK-FAC 把 Fisher 近似为两个协方差矩阵的 Kronecker 积，并对其特征值做显式校正。两个因子定义为：

$$A_{W_2} = \mathbb{E}_{y \sim \pi_\theta(\cdot|x)}\!\left[ m^\ell (m^\ell)^\top \right] \in \mathbb{R}^{d_\text{mlp} \times d_\text{mlp}}$$

$$S_{W_2} = \mathbb{E}_{y \sim \pi_\theta(\cdot|x)}\!\left[ \delta\, \delta^\top \right] \in \mathbb{R}^{d \times d}$$

朴素的 Kronecker 近似 $F_{W_2} \approx S_{W_2} \otimes A_{W_2}$ 假设特征值为 $\lambda_{S,i} \cdot \lambda_{A,j}$，在有限样本下不准确。EK-FAC 保留特征向量基，但对特征值做显式校正。

对 $A_{W_2}$ 和 $S_{W_2}$ 各做特征分解：

$$A_{W_2} = U_A \Sigma_A U_A^\top, \quad S_{W_2} = U_S \Sigma_S U_S^\top$$

在 Kronecker 特征基 $U_S \otimes U_A$ 下，用 Monte Carlo 估计每个方向的真实曲率：

$$\Lambda_\text{corrected} = \mathbb{E}\!\left[\left(U_A^\top \nabla_{W_2}\mathcal{L} \cdot U_S\right)^{\odot 2}\right]$$

即把每个样本的梯度矩阵投影到 Kronecker 特征基后逐元素平方，再对样本平均。

#### 加 Damping 后的 IHVP 公式

对向量 $v$（probe 梯度 flatten 成向量），设 $G = v.\text{reshape}(d, d_\text{mlp})$：

$$\hat{F}_{W_2}^{-1} v = \text{vec}\!\left( U_S \cdot \frac{U_S^\top G\, U_A}{\text{denom}} \cdot U_A^\top \right)$$

**denom 的计算是两级自适应的**（MDA `ekfac_blocks.py` 实际实现）：

$$\text{denom}_i = \max\!\left(\Lambda_i + \alpha \cdot \bar{\Lambda},\;\; \lambda_\text{floor}\right)$$

- $\alpha = 0.1$（`damping_alpha`）：相对 damping，自适应于曲率量级
- $\lambda_\text{floor} = 10^{-5}$（`damping`）：绝对下界

#### 与 MDA 实现的对应关系

| 步骤 | MDA 函数 | 作用 |
|------|---------|------|
| Stage 1A | `stage1A_accumulate_AS` | 累积 $A$、$S$，做特征分解，得 $U_A$、$U_S$ |
| Stage 1B | `stage1B_fit_lambda` | 在 Kronecker 基底下估计 $\Lambda_\text{corrected}$ |
| IHVP | `ekfac_blocks.inverse_hvp` | 给定 probe gradient $v$，输出 $p = \hat{F}^{-1}v$ |
| Phase 2 | `influence_phase2.phase2_score_*` | 对每条训练样本算 $-g_i^\top p$，复用 $p$ |

---

## 14. EK-FAC 如何接到这个问题上？

你的目标不是直接复用 supervised IF，而是复用 EK-FAC 的 Fisher inverse-vector product 机制。

具体流程：

1. 选择参数子空间 $\phi$，本项目固定为 Layer 19 MLP 的 $W_2$。
2. 用 PPO 后模型 $\pi_{\theta^*}$ 对一批 prompt 采样 rollout。
3. 对这些 rollout 计算 sequence logprob。
4. 对 $\phi$ 求 gradient，得到 $s_\phi(x,y)$。
5. 用这些 score/activation 估计 EK-FAC factors。
6. 对 eval 样本计算 $g_{\mathrm{eval},\phi}$。
7. **先对 eval gradient 做一次 IHVP**，得到：

$$
p = F_{\mathrm{EKFAC},\phi}^{-1}\, g_{\mathrm{eval},\phi}
$$

8. 对每条训练 rollout $m$，只做一次点积：

$$
\mathcal I_m = -p^\top s_{m,\phi}
$$

> **为什么必须先算 $p$ 再对训练样本做点积**：eval/probe 样本通常只有少量（几条），IHVP 只做一次；训练 rollout 可能有数万条，每条只需做向量点积。若反过来对每条训练样本做 IHVP，代价是 $M$ 倍 IHVP，完全不可行。

---

### 工程注记：EK-FAC 完整计算流程（基于 MDA 实现）

#### Stage 1A：累积 $A$、$S$ 并做特征分解

用从 $\pi_\theta$ **采样的 pseudo label**（不用真实 label）累积 Kronecker 因子。

**什么是 pseudo label**：不用训练数据的真实下一个 token，而是让模型从当前输出分布 $\pi_\theta$ 里随机采一个 token 作为 label。这对应 Fisher 定义里期望 $\mathbb{E}_{y \sim \pi_\theta}$ 的正确实现，用真实 label 算出的是 empirical Fisher，对 RL setting 不适用。

**数据来源**：用和训练 rollout 相同的 prompt 集合，随机抽 $K$ 个 prompt 即可，不需要特别挑选。

```python
# stage1A_accumulate_AS
logits = model(input_ids)                          # [B, T, V]
sampled_labels = compute_pseudo_labels(logits)     # 从 π_θ 采样 pseudo label

# RL setting：prompt 部分不参与梯度计算
full_labels = torch.cat([
    torch.full((T_prompt,), -100, dtype=torch.long),  # prompt：忽略
    sampled_labels[T_prompt:]                           # response：pseudo label
])
loss = F.cross_entropy(
    logits.reshape(-1, V).float(),
    full_labels.reshape(-1),
    ignore_index=-100,
    reduction="sum"
)
# 拿到激活 m 和梯度 delta，累积 A, S
A_accum += m.T @ m
S_accum += delta.T @ delta
token_count += T_response   # 只数 response token 数

# 全部累积完之后：
A = A_accum / token_count   # per-token 归一化
A = 0.5 * (A + A.T)
_, Q_A = torch.linalg.eigh(A)   # 特征向量
# S 同理 → Q_S
```

#### Stage 1B：拟合校正特征值 $\Lambda_\text{corrected}$

在 Stage 1A 得到的 $U_A$、$U_S$ 基底下，再过一遍同一批 prompt（同样用 pseudo label），估计真实特征值：

```python
# stage1B_fit_lambda
dW = m.T @ delta            # 梯度矩阵 (d_mlp, d)
ge = Q_A.T @ dW @ Q_S      # 投影到 Kronecker 特征基
lambda_sum += ge.pow(2)     # 逐元素平方，累积
weight_sum += B             # 除的是 sequence 数（不是 token 数）

Lambda_corrected = (lambda_sum / weight_sum).flatten()
```

#### Stage 2：算 probe/eval gradient，立即做 IHVP

```python
# 算 eval 侧的梯度 g_eval（per-token 平均）
# 同样需要 prompt mask
g_eval = compute_eval_grad(model, eval_batch, ignore_prompt=True)

# 立即做 IHVP，得到 p（只算这一次！）
ge = Q_A.T @ g_eval @ Q_S
denom = Lambda + damping_alpha * Lambda.mean()   # 自适应 damping
denom = torch.clamp(denom, min=damping)          # 绝对下界 1e-5
p = Q_A @ (ge.flatten() / denom).reshape(d_mlp, d) @ Q_S.T
# p 对所有训练样本复用
```

#### Stage 3：每条训练样本算 score（只做点积）

> **重要约束：Stage 3 必须 batch_size=1。** IF 定义在单条样本上，若 batch 里有多条序列，`autograd.grad` 拿到的是 $\sum_i g_i$，无法还原单条样本的梯度。

```python
# 每次只喂一条训练 rollout，batch_size=1
# 用真实记录的 rollout label（不是 pseudo label）
# RL setting：prompt 部分 mask 掉
full_labels = torch.cat([
    torch.full((T_prompt,), -100, dtype=torch.long),
    recorded_response_ids
])
loss = F.cross_entropy(
    logits.reshape(-1, V).float(),
    full_labels.reshape(-1),
    ignore_index=-100,
    reduction="mean"    # RL variable-length rollout 用 mean（见 Section 18.6）
)
g = autograd.grad(loss, W_2)[0]   # (d_mlp, d)

score = -(g * p).sum().item()     # 点积，标量
```

**Stage 3 用真实 label，Stage 1 用 pseudo label**：Stage 1 估计曲率（需要 $\pi_\theta$ 的期望），Stage 3 算训练样本的实际影响（用观测到的数据）。

#### 整体数据流

```
Stage 1A（一次性，pseudo label，可 batch）：
  K 个 prompt → 采样 response → 累积 A, S（per-token 归一化）
  → 特征分解 → Q_A, Q_S

Stage 1B（一次性，pseudo label，可 batch）：
  同一批 prompt → 在 Q_A ⊗ Q_S 基底下估计 → Lambda_corrected

Stage 2（每个 eval 样本，通常很少）：
  eval → 算 g_eval（per-token mean，prompt mask）
  → p = inverse_hvp(g_eval)    ← IHVP 在这里，只算一次

Stage 3（每条训练 rollout，可能很多，batch_size=1）：
  z_i → 真实 label，per-token mean，prompt mask
  → g_i = ∇_φ L(z_i)
  → score_i = -(g_i * p).sum() ← 只是点积
```

---

### RL Setting 与 MDA 的两个关键差异

#### 差异一：Prompt Mask（RL 独有，MDA 没有）

MDA 做 pretraining，整条序列都是模型生成的内容，不存在 prompt/response 区分。你的 RL setting 里，**prompt 部分不是 $\pi_\theta$ 生成的**，不应参与 $\log\pi_\theta(y|x)$ 的计算。这个 mask 在 Stage 1A/1B 和 Stage 3 里都必须加（见上方代码中的 `ignore_index=-100`）。

漏掉这个 mask 的后果：prompt token 的曲率被错误地计入 Fisher，score 失去对 rollout 质量的分辨力，且极难 debug。

#### 差异二：Stage 3 的 reduction（MDA 用 sum，RL 用 mean）

MDA 的 `influence_phase2.py` 里 Stage 3 用 `reduction="sum"`，这没问题，因为他们的所有序列都截到同一个固定长度，sum 和 mean 只差一个常数，不影响 ranking。

你的 RL setting 里 rollout 长度可变，用 `sum` 会导致长 rollout 系统性排名靠前，与 rollout 实际质量无关。应改为 `reduction="mean"`，与 Stage 1A/1B 的 per-token 归一化对称。

---

## 15. Toxic Vector / Probe 如何接入？

如果只关心 eval logprob，可以设：

$$
f(\theta)
=
\log\pi_\theta(y_{\mathrm{eval}}|x_{\mathrm{eval}})
$$

如果关心 toxic vector / probe，也可以把 $f$ 定义为某层 hidden state 与 toxic direction 的投影：

$$
f(\theta)
=
v_{\mathrm{toxic}}^\top
h_\ell(x_{\mathrm{eval}},y_{\mathrm{eval}};\theta)
$$

或者对一批 toxic prompts：

$$
f(\theta)
=
\frac{1}{N}
\sum_{i=1}^{N}
 v_{\mathrm{toxic}}^\top
 h_\ell(x_i,y_i;\theta)
$$

然后 influence 仍然是：

$$
\mathcal I_f(m)
=
-
\nabla_\phi f(\theta^*)^\top
F_\phi^{-1}
s_{m,\phi}
$$

区别只在于 $\nabla_\phi f$ 的定义换了。

建议第一版仍然使用 eval logprob，因为它最简单、最干净、最容易验证。

---

## 16. 实验路线

执行计划与 `CLAUDE.md` 的 Phase Structure 一致。本节给出每个 phase 的研究意义。

### Phase 1：Effective Reward Variance 诊断

目标：检查 PPO 后 policy 是否接近 KL-RL 最优点，从而判断 Section 5 的核心相消假设是否成立。

流程：选若干训练 prompt $x$；从 PPO 后模型采样 $K$ 条 rollout；对每条计算 reward、policy logprob、reference logprob、effective reward；计算同一 prompt 内部的 effective reward variance；对多个 prompt 重复。

判断：variance 小 → 理论近似可信；variance 大 → policy 还没接近 KL-RL 最优结构，IF score 数值上算得出来但解释上需要保留。

### Phase 2：EK-FAC Influence Function

目标：对 PPO rollout 计算 rollout-level influence score，识别哪些 rollout 对 eval logprob 贡献最大。

流程：固定 prompt 和 eval 样本；只看 Layer 19 MLP $W_2$；先用 toy transformer 做数值正确性测试（与直接求逆的 ground truth Fisher 对比，1e-3 容差）；通过后在真实 GPT2-medium 上跑 EK-FAC pipeline；同时计算 gradient similarity 作为 sanity baseline。

输出：每条 rollout 的 IF score 和排名。

### Phase 3：Mechanistic 验证

目标：把 IF 排名连接到 Lee et al. 标注的 toxic value vectors 上，验证 IF 抓到的是不是真正驱动 detoxification 的 rollout。

流程：取 IF 排前 k 和后 k 的 rollout；测它们在 PPO checkpoint 上对 $\text{MLP.v}^{19}_{770}$ 的 activation $m^{19}_{770} = \sigma(x^{19}\cdot k^{19}_{770})$；计算 IF score gradient 方向与 Lee et al. 的 $\delta_x$（PPO 与 ref model 的 residual stream 差）的 cosine similarity；检查 top-k rollout 文本内容是否在语义上和 detoxification 行为一致。

---

## 17. 需要记录的关键诊断量

每个实验最好记录：

1. prompt 内部 effective reward variance；
2. 原始 reward variance；
3. KL penalty variance；
4. reward 与 KL penalty 的相关性；
5. $\|s_m\|$；
6. $\|g_{\mathrm{eval}}\|$；
7. $g_{\mathrm{eval}}^\top s_m$；
8. $g_{\mathrm{eval}}^\top F^{-1}s_m$；
9. damping $\lambda$；
10. PPO 的 $\beta$；
11. rollout length（response 部分）；
12. reward model score；
13. sequence-level KL；
14. token-average KL。

这些诊断量可以帮助判断 influence score 准不准，以及不准时是哪个环节坏了。

---

## 18. 当前理论最需要强调的假设

### 18.1 不能直接扔掉 $\mathbb E_x$

完整理论应该保留 $\mathbb E_{x\sim\mathcal D}$。单 prompt 版本只是局部分析，不是完整目标的替代。

### 18.2 $\pi_{\theta^*}$ 需要接近 KL-RL 最优族

核心相消依赖 $\tilde R(x,y,\theta^*) \approx \text{constant in } y$，需要用 effective reward variance 做诊断。

### 18.3 Fisher 需要 damping

真实模型里 Fisher 往往奇异或病态，实际公式应该写成：

$$
\mathcal I_f(m)
=
-
\nabla_\phi f^\top
(F_\phi+\lambda I)^{-1}
s_{m,\phi}
$$

> **MDA 实际实现的 Damping 机制**（`ekfac_blocks.py`）：两级自适应：
>
> $$\text{denom}_i = \max\!\left(\Lambda_i + \underbrace{0.1 \cdot \bar{\Lambda}}_{\text{相对项，自适应}},\;\; \underbrace{10^{-5}}_{\text{绝对下界}}\right)$$
>
> - `damping_alpha = 0.1`：相对项，自适应于曲率量级，不需要跨层/跨模型重新调参。
> - `damping = 1e-5`：绝对下界，`torch.clamp` 保证分母不为零。
>
> **选取建议**：`damping_alpha = 0.1` 通常不需要调整。若出现数值爆炸可将 `damping` 上调至 $10^{-4}$；若 ranking 对 damping 非常敏感，说明 $\Lambda$ 中有大量接近零的特征值，应首先检查 Stage 1A/1B 累积的样本数量是否足够。

### 18.4 Influence 是参数子空间内的局部近似

如果只看 某些 MLP 层，结论应该表述为：

> 在选定参数子空间中，该 rollout 对 eval 目标的近似影响。

不能声称这是全模型完整影响。

### 18.5 PPO rollout 是 sequence-level 样本

训练点不是单 token，而是 $(x,y_{1:T})$。score 是整条 response 序列 token logprob gradient 的和（数学定义）或平均（工程实现，见 Section 18.6）。

### 18.6 序列长度偏差与 Per-Token 归一化

**问题**：原始 score 定义为整条序列的梯度之和，norm 与序列长度 $T$ 正相关。较长的 rollout 的 $\|s_{m,\phi}\|$ 系统性偏大，Fisher inverse 会部分补偿但无法完全消除。

**MDA 的处理**：MDA（Chen et al., 2026）所有训练序列都截到同一个固定长度 `seq_length`，因此 `reduction="sum"` 和 `"mean"` 只差常数，不影响 ranking，没有长度偏差问题。这不是因为他们用了 mean，而是因为长度固定。

**RL setting 的修复**：rollout 长度可变，需显式归一化：

$$s_{m,\phi} = \frac{1}{T}\sum_{t=1}^{T} \nabla_\phi \log\pi_{\theta^*}(y_{m,t}|x, y_{m,<t})$$

$$g_{\text{eval},\phi} = \frac{1}{L}\sum_{t=1}^{L} \nabla_\phi \log\pi_{\theta^*}(y_{\text{eval},t}|x_{\text{eval}}, y_{\text{eval},<t})$$

实现：loss 的 `reduction` 改为 `'mean'`，两侧必须同时归一化。

---

## 19. 推荐的执行顺序

### 第一阶段：理论整理

写清楚单 prompt 版本；$\mathbb E_x$ 版本；sequence-level score；eval logprob 目标；Fisher/damped Fisher；参数子空间 $\phi$。

### 第二阶段：最小数值验证

选小模型；少量 prompt；采样 rollout；计算 effective reward variance；计算 diagonal Fisher influence；验证 influence ranking 是否有意义。

### 第三阶段：小规模 PPO 验证

用 PPO 后模型；固定 reward model 和 ref model；计算 rollout influence；做 upweight/downweight/remove；重新训练或局部更新；比较 IF 预测和真实 eval logprob 变化。

### 第四阶段：EK-FAC 扩展

接入 EK-FAC / K-FAC factor 估计；替换 diagonal Fisher inverse；和 baseline 对比；检查是否提升 retraining correlation。

### 第五阶段：Mechanistic / Toxic Probe 结合

选择 toxic vector 或 probe；定义 probe-based $f$；计算 $\nabla_\phi f$；用同一套 Fisher influence 公式找高影响 rollout；检查这些 rollout 是否在语义上与 toxic behavior 相关。

---

## 20. 最终可以形成的研究主张

如果实验成功，可以形成如下主张：

> 在 KL-regularized RLHF-PPO 中，若训练后的 policy 在局部接近 KL-RL 最优解，则目标梯度的 Jacobian/Hessian 结构可以近似由 Fisher 控制。因此，PPO rollout 对 eval logprob 或 probe score 的影响可以用 natural-gradient 形式的 influence score 估计：

$$
\mathcal I_f(m)
=
-
\nabla_\phi f(\theta^*)^\top
(F_\phi+\lambda I)^{-1}
\nabla_\phi\log\pi_{\theta^*}(y_m|x_m)
$$

其中 rollout 是完整 sequence-level 样本，Fisher 可以在选定参数子空间中用 diagonal Fisher、K-FAC 或 EK-FAC 近似。

这个方法可以用于：

1. 找出最影响某个 eval completion 的 PPO rollout；
2. 分析 RLHF 后模型行为改变来自哪些训练样本；
3. 连接 influence function 与 mechanistic probe；
4. 解释某些 toxic / refusal / helpfulness 行为的训练来源；
5. 为 RLHF 数据清洗和 debug 提供工具。


## 附录 A：模型选择历史与备选方案

### 最终选择：GPT-Neo-125M

经过对 GPT2-medium 上 7 轮 PPO 调试失败的诊断，确定根因是
GPT2-medium 作为 raw pretrain model（无 SFT）在 toxic token 上的
base log-prob 过高，使任何 KL estimator 在 detoxification 训练中
不稳。最终选择 GPT-Neo-125M：

- TRL 0.9.6 官方 detoxification tutorial 直接基于此模型
- 单 GPU PPO 训练 ~3 小时，迭代快
- Probe.pt 需要在此模型上重训（架构 768-dim, 12 layer），但流程
  与 Lee et al. Section 3.1 完全一致

### 备选方案：OLMo 2 1B SFT (allenai/OLMo-2-0425-1B-SFT)

如果时间充裕，在 Phase 4 之后可考虑迁移到 OLMo 2 1B SFT 上重做
全流程：

- 真正经过 Tülu 3 SFT 对齐（不是单纯 pretrain）
- 1.48B 参数，48 GB 显存下用 gradient checkpointing 可训
- 论文 narrative 升级："我们的 IF 分析在 SFT-aligned modern model
  上同样适用"

要求：transformers 升级到 >= 4.48，验证 TRL 0.9.6 兼容性，
预计 PPO 训练 ~12-15 小时单次。

未做 OLMo 3 7B 因为单 GPU PPO 训练硬件不足（即使 48 GB）。