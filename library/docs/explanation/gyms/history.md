# HistoryGym

`HistoryGym` wraps a simulation gym and supplies a fixed-length observation history to policies that expect history to be
provided externally.

Selected camera tensors change from `[B, C, H, W]` to `[B, T, C, H, W]`, and selected state tensors change from `[B, D]` to
`[B, T, D]`, where `T = n_obs_steps`. Unselected observation fields remain unchanged.

## Policy Compatibility

**Do not use `HistoryGym` with policies that manage their own observation history.**

Accepting a five-dimensional image tensor does not, by itself, make a policy compatible. The history length, temporal
sampling, ordering, and padding must match the policy's expectations.

## Padding and Ordering

History is ordered from oldest to newest and always includes the current observation. `n_obs_steps` controls the number of
observations returned; `stride` controls the spacing between them in environment steps. Both must be positive integers.

History advances on every environment step, not just when the policy runs inference. The returned offsets are
`[-(n_obs_steps - 1) * stride, ..., -stride, 0]`. The wrapper retains `(n_obs_steps - 1) * stride + 1` observations internally.

On reset, the initial observation fills any missing history. With the default `stride=1`, the window contains consecutive
observations.

For `n_obs_steps=4`:

```text
reset:  [obs0, obs0, obs0, obs0]
step 1: [obs0, obs0, obs0, obs1]
step 2: [obs0, obs0, obs1, obs2]
step 3: [obs0, obs1, obs2, obs3]
step 4: [obs1, obs2, obs3, obs4]
```

Selected values retain the time axis even when `n_obs_steps=1`.

For `n_obs_steps=4, stride=2`, the offsets are `[-6, -4, -2, 0]`. The window still updates on every step:

```text
reset:  [obs0, obs0, obs0, obs0]
step 1: [obs0, obs0, obs0, obs1]
step 2: [obs0, obs0, obs0, obs2]
step 3: [obs0, obs0, obs1, obs3]
step 4: [obs0, obs0, obs2, obs4]
step 5: [obs0, obs1, obs3, obs5]
step 6: [obs0, obs2, obs4, obs6]
step 7: [obs1, obs3, obs5, obs7]
```

## Usage

Select the observation fields to stack using dotted keys:

```python
from physicalai.gyms import HistoryGym, LiberoGym

gym = HistoryGym(
    LiberoGym(task_suite="libero_10", task_id=0),
    n_obs_steps=4,
    keys={"images.image", "images.image2", "state"},
    stride=2,
)
```

The equivalent YAML configuration is:

```yaml
class_path: physicalai.gyms.HistoryGym
init_args:
  gym:
    class_path: physicalai.gyms.LiberoGym
    init_args:
      task_suite: libero_10
      task_id: 0
  n_obs_steps: 4
  stride: 2
  keys:
    - images.image
    - images.image2
    - state
```
