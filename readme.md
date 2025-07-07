## Score Entropy Discrete Diffusion Language Model

### Diffusion language models

Diffusion models learn a generative reverse process by inverting a fixed forward noising process. Language models based on this idea can benefit from properties of diffusion: sampling is parallelizable, and the generation process can be conditioned.
For example, one can initialise the process from a partial sequence and sample infillings consistent with a learned distribution. They are also able to learn structural constraints, which can be benefitial for certain domains where global consistency is important, such as source code.

### Implementation overview

This repo contains a self-contained, mostly from-scratch reimplementation of the Score Entropy Discrete Diffusion (SEDD) model from [Lou et al. (2023)](https://arxiv.org/abs/2310.16834).

This implementation focuses on clarity, and implements the forward process specialized to an absorbing transition matrix. Sampling of a random timestamp, perturbation of the sequence, and other computations needed to evaluate the integral in the objective are part of the loss function, which can be found in `loss.py`.

`reverse.py` implements a batched version of the Tweedie $\tau$-leaping denoising algorithm (Alg. 2 in the paper). It is optimized for small vocabulary sizes using dense matrices.

The encoder-only transformer in the score network is significantly simplified, using a sinusoidal positional embedding and a simple MLP time embedding (`score.py`). It takes $(x_t, \overline{\sigma(t)})$ as input and outputs unnormalized log densities corresponding to scores for all possible states (incuding the mask) for each position in the sequence.

The same log-linear noise schedule from the original implementation is used, such that $\sigma(t)$ increases monotonically. The score network is also parametrised with the total noise level $\overline{\sigma(t)}$ instead of the timestamp $t$.

### Mathematical setup for the SEDD


Let $X_t$ be a continuous-time Markov process on token sequences with time-dependent transition matrix $Q_t$. Under suitable regularity conditions, the transition kernel is given by

$$
\mathbb{P}(X_t = y \mid X_0 = x) = \left[\exp\left(\int_0^t Q_\tau\ d\tau\right)\right]_{y,x}
$$

We define $Q_t = \sigma(t) Q^{\text{absorb}}$, where $\sigma(t)$ is a monotonically increasing noise rate and $Q^{\text{absorb}}$ is

$$
Q^{\text{absorb}}(a, b) =
\begin{cases}
-1 & \text{if } a = b < V \\
1 & \text{if } a = V \text{ and } b < V \\
0 & \text{else}
\end{cases}
$$

Let $\overline{\sigma}(t) = \int_0^t \sigma(\tau)\,d\tau$. Then, for a token $x \in \{1, \dots, V-1\}$ and absorbing token $V = \text{mask}$, the marginal at time $t$ satisfies

$$
\mathbb{P}(X_t = \text{mask} \mid X_0 = x) = 1 - \exp[-\overline{\sigma}(t)]
$$

and all mass accumulates in the absorbing state as $t \to \infty$: $\mathbb{P}(X_t = \text{mask} \mid X_s = \text{mask}) = 1$.

The model is trained to estimate unnormalized transition ratios (or, _scores_) $\frac{p_t(y)}{p_t(x)}$, quantifying how likely y is as a reverse-time denoising candidate given current
state x.

For the loss function, defining $Q_t: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ as:

$$
Q_t(x_1x_2...x_V, x'_1x'_2...x'_V) := \begin{cases}
0 & \text{if } x \text{ and } x'\text{ have Hamming distance } \neq 1 \\
Q^{\text{absorb}}(x_i,x'_i) & \text{else if } x_i \neq x'_i
\end{cases}
$$

this gives a transition operator on sequences where only one nonabsorbing token changes to the absorbing token at each step. The process cannot transition directly between nonabsorbing tokens: instead, all transitions pass through the absorbing state.

Let $x_0$ be the clean sequence, $x_t$ the noised sequence at time $t$, and $s_\theta(x_t, \overline{\sigma}(t))$ the model output. The objective penalises divergence between the predicted score
and the target ratio of transition probabilities derived from the forward process:

```math
\int_0^T \mathbb{E}_{x_t \sim p_t(\cdot \mid x_0)}
\sum_{i : (x_t)_i = V} \sum_{b < V}
\left[
s_\theta(x_t, t)[x_{/i} \leftarrow b]
- \frac{p(b \mid (x_0)_i)}{p((x_t)_i \mid (x_0)_i)} \log (s_\theta(x_t, t)[x_{/i} \leftarrow b])
+ K\left(\frac{p(b \mid (x_0)_i)}{p((x_t)_i \mid (x_0)_i)}\right)
\right]
```

where $x_{/i} \leftarrow b$ denotes replacing position $i$ with token $b$, and $K(\cdot)$ is a normalising term.

At inference time, we use $\tau$-leaping denoising to approximate the reverse process. Starting from a fully masked sequence $x_t$ at time $t$, time will flow backwards and, using the learned prediction of $s_\theta$ for what the absorbing states should be replaced with to match the true data distribution, $x_t$ will be _denoised_.

The reverse transition probability for each position $i$ and candidate token $y$ used during denoising is given by:

```math
p(y \mid x_t^i) =
\left[
\exp\left((\overline{\sigma}(t - \Delta t) - \overline{\sigma}(t)) Q^{\text{absorb}}\right)
s_\theta(x_t, \overline{\sigma}(t))_i
\right]_y
\cdot
\left[
\exp\left((\overline{\sigma}(t) - \overline{\sigma}(t - \Delta t)) Q^{\text{absorb}}\right)
\right]_{x_t^i, y}
```

Since $Q^{\text{absorb}}$ has a simple structure, the matrix exponential has a closed form:

$$
\exp(a Q^{\text{absorb}}) =
\begin{pmatrix}
e^{-a} \cdot I & 0 \\
(1 - e^{-a}) \cdot 1^T & 1
\end{pmatrix}
$$

making the process computationally tractable.

### Evaluation

We apply the model to the ACYP protein dataset - credit for this idea goes to [Alex Carlin](https://alexcarlin.bearblog.dev/score-entropy-discrete-diffusion-models-for-protein-design/). The dataset consists of character-level protein sequences over a 21-token alphabet, with sequence length capped at 127. Special start and end tokens are added.

Training was done for 30k steps on a single A100 GPU. Sampling used 1024 denoising steps. Folding of sampled sequences was performed using ESMFold to evaluate plausibility. Folding success was low (14 out of 300), but all successful structures were syntactically correct, suggesting the model learns the correct structural priors even without explicit folding success as part of the objective.

An example of a generated protein:
<p align="center">
<img src="https://raw.githubusercontent.com/mstarodub/dllm/refs/heads/master/figures/proteins/27400.png" width="360" height="266" >
</p>

We also attempted to apply the model to the TinyStories dataset. This is currently broken for some reason (patches are welcome).
