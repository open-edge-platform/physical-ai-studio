# Action Heads

Action heads are shared components that turn a conditioning context into a
chunk of actions `(B, chunk_size, action_dim)`.

```python
from physicalai.policies.components import ActionHead, DiffusionActionHead, IterativeActionHead
```

- `ActionHead` defines `compute_loss(actions, context)` and `sample(context)`.
- `IterativeActionHead` adds the denoising loop shared by flow matching and
  diffusion. Subclasses implement `denoise`, `timesteps` and `step`.
- `DiffusionActionHead` is a ready-made `IterativeActionHead` for DDPM / DDIM
  diffusion. Subclasses only implement `denoise`. See
  [Diffusion Action Head](#diffusion-action-head).
- None of these classes hold weights, so existing policies can adopt them
  without changing their checkpoint keys.

## Diffusion Action Head

`DiffusionActionHead` implements the denoising diffusion used by
[Diffusion Policy](https://arxiv.org/abs/2303.04137) (Chi et al. 2023). It owns
the noise schedule, the training loss and the sampler. A policy supplies only
`denoise`, the network that predicts the noise, and the context it conditions
on. The [diffusion head README](../../../src/physicalai/policies/components/action_heads/diffusion/README.md) has the arguments, a runnable tiny
head, and measured timings, dtypes and graph replay.

### Why diffusion for actions

Demonstrations are often multimodal: to avoid an obstacle, an operator goes
left half the time and right the other half. A regression head minimizes the
mean squared error, so it predicts the average, straight into the obstacle.
A diffusion head learns the whole action distribution and samples one of the
modes.

### Forward process

Actions are normalized to `[-1, 1]`. Training corrupts a clean chunk $x_0$
with Gaussian noise over $T$ steps (`num_train_timesteps`). Step $t$ adds
noise with variance $\beta_t$:

$$q(x_t \mid x_{t-1}) = \mathcal{N}\left(\sqrt{1 - \beta_t}\, x_{t-1},\; \beta_t I\right)$$

With $\alpha_t = 1 - \beta_t$ and $\bar\alpha_t = \prod_{s \le t} \alpha_s$,
the steps compose into a closed form, so training can jump to any $t$ directly:

$$x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)$$

$\sqrt{\bar\alpha_t}$ scales the signal and $\sqrt{1 - \bar\alpha_t}$ scales the
noise. At $t = 0$ the chunk is almost clean. At $t = T - 1$ it is almost pure
noise. `add_noise(actions, noise, t)` computes this expression.

The schedule controls how fast $\bar\alpha_t$ falls. `make_betas` provides the
`diffusers` definitions:

| `beta_schedule`     | Definition                                                                                          |
| ------------------- | --------------------------------------------------------------------------------------------------- |
| `linear`            | $\beta_t$ evenly spaced from `beta_start` to `beta_end`                                             |
| `scaled_linear`     | $\sqrt{\beta_t}$ evenly spaced, then squared                                                        |
| `squaredcos_cap_v2` | $\bar\alpha(s) = \cos^2\left(\frac{s + 0.008}{1.008} \cdot \frac{\pi}{2}\right)$, $\beta \le 0.999$ |

The cosine schedule (`squaredcos_cap_v2`, the default) removes information more
slowly at the start than the linear one.

### Training

`compute_loss(actions, context)` draws a random $t$ per sample, noises the
actions and regresses the target:

$$\mathcal{L} = \left\lVert \epsilon - \epsilon_\theta(x_t, t, c) \right\rVert^2$$

```python
noise = torch.randn_like(actions)
t = torch.randint(0, T, (B,))
x_t = add_noise(actions, noise, t)
loss = (denoise(x_t, t, context) - noise) ** 2  # (B, chunk_size, action_dim), unreduced
```

With `prediction_type="sample"` the network predicts $x_0$ instead and the
target is `actions`. The loss is returned per element, so the policy can mask
padded action steps with `reduce_losses`.

### Sampling

Both prediction types give an estimate of the clean chunk:

$$\hat x_0 = \frac{x_t - \sqrt{1 - \bar\alpha_t}\, \hat\epsilon}{\sqrt{\bar\alpha_t}}$$

With `clip_sample=True`, $\hat x_0$ is clamped to
`[-clip_sample_range, clip_sample_range]`. The noise estimate is then
recomputed from the clamped $\hat x_0$, so the two terms of the update agree.
Each step moves from $t$ to an earlier $t'$ with the DDIM update
(Song et al. 2020):

$$x_{t'} = \sqrt{\bar\alpha_{t'}}\, \hat x_0 + \sqrt{1 - \bar\alpha_{t'} - \sigma^2}\, \hat\epsilon + \sigma z, \qquad z \sim \mathcal{N}(0, I)$$

$$\sigma^2 = \eta^2 \, \frac{1 - \bar\alpha_{t'}}{1 - \bar\alpha_t} \left(1 - \frac{\bar\alpha_t}{\bar\alpha_{t'}}\right)$$

The first term is the current estimate of the data. The second keeps part of
the predicted noise. The third adds fresh noise. `eta` selects the sampler:

| `eta` | Sampler | Behavior                                                                |
| ----- | ------- | ----------------------------------------------------------------------- |
| `1.0` | DDPM    | $\sigma$ is the true posterior variance. Stochastic, usually `T` steps. |
| `0.0` | DDIM    | No fresh noise. Deterministic, good results with about 10 steps.        |

### How it fits `IterativeActionHead`

The base class runs the sampling loop and `DiffusionActionHead` fills in each
part:

| Method         | Diffusion implementation                                             |
| -------------- | -------------------------------------------------------------------- |
| `timesteps`    | Integers `T - 1` down to `0`, then `-1`, e.g. `[90, 80, ..., 0, -1]` |
| `step`         | The DDIM update above (DDPM when `eta = 1`)                          |
| `compute_loss` | Noise-prediction MSE                                                 |
| `denoise`      | Abstract. The policy's network: `(x_t, t, context) -> ε̂ or x̂₀`       |

`-1` stands for the clean data, where $\bar\alpha = 1$. The schedule buffers
store a leading $\bar\alpha = 1$, so timestep `t` is read at index `t + 1`.
The lookup uses `index_select`, never `buffer[t]`: indexing with a 0-d tensor
calls `.item()`, which syncs with the device and breaks
[graph replay](../../../src/physicalai/policies/components/action_heads/diffusion/README.md#graph-replay).

The defaults (`T = 100`, cosine schedule, noise prediction, clipping, DDPM)
match LeRobot's Diffusion Policy. The unit tests check the maths with a
perfect denoiser: a DDIM step lands exactly on the forward process at the next
timestep, a DDPM step samples the posterior $q(x_{t-1} \mid x_t, x_0)$, and
both samplers recover $x_0$.

### Dtypes

bf16 has 7 mantissa bits, so it cannot represent $\bar\alpha$ close to 1:

| Value                           | Exact  | bf16     | fp16   |
| ------------------------------- | ------ | -------- | ------ |
| $1 - \bar\alpha_0$ (`T = 100`)  | 6.3e-4 | **0**    | 4.9e-4 |
| $1 - \bar\alpha_0$ (`T = 1000`) | 4.1e-5 | **0**    | **0**  |
| timestep `999`                  | 999    | **1000** | 999    |

The update divides by $\sqrt{1 - \bar\alpha_t}$, so computing it from a rounded
$\bar\alpha$ produces infinities, and a rounded timestep reads the wrong
schedule entry. The head avoids both without any float32 promotion:

- It stores $\sqrt{\bar\alpha}$ and $\sqrt{1 - \bar\alpha}$ directly, so
  $1 - \bar\alpha$ is never computed at runtime. $\sqrt{1 - \bar\alpha_0}$ is
  0.025 at `T = 100` and 0.006 at `T = 1000`, which bf16 and fp16 represent
  well.
- The schedule is built in float32, whatever the default dtype. After that
  its buffers follow the module like any buffer, so `head.to(torch.bfloat16)`
  runs the network, the schedule and `step` all in bf16.
- Timesteps are `torch.long`. Convert them to float32 before scaling inside
  `denoise`, as the [example head](../../../src/physicalai/policies/components/action_heads/diffusion/README.md#define) does.

Both low precisions work. fp16 has 10 mantissa bits to bf16's 7 and is about
10x more accurate on the example head, but its range stops at 65504, so a
large model trained in bf16 can overflow in fp16. Check a model's activations
before choosing fp16.
