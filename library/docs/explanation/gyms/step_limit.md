# StepLimit

A wrapper that enforces a maximum number of steps per episode.

`StepLimit` tracks environment steps and forces truncation once `max_steps` is reached.
Useful for tasks without natural termination, or when controlling rollout length.

```mermaid
classDiagram
    class StepLimit {
        + gym : Gym
        + max_steps : int
        + step_count : int
        --
        + __init__(gym: Gym, max_steps: int)
        + reset(seed: int|None, **kwargs) Observation, dict|list[dict]
        + step(action) Observation, float|list[float], bool|list[bool], bool|list[bool], dict|list[dict]
        + render(...) Any
        + close() None
        + sample_action() torch.Tensor
        + to_observation(raw_obs) Observation
    }

    StepLimit --> Gym : wraps
```

Example:

```python
from physicalai.gyms import GymnasiumGym
from physicalai.gyms.step_limit import StepLimit

env = GymnasiumGym("CartPole-v1")
env = StepLimit(env, max_steps=200)

obs, info = env.reset()
action = env.sample_action()
obs, reward, terminated, truncated, info = env.step(action)
```

or with our helper function:

```python
from physicalai.gyms.step_limit import with_step_limit
env = with_step_limit(env, max_steps=200)
```
