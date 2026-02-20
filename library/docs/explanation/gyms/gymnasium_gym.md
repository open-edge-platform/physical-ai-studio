# Gymnasium Gym

Adapter providing unified Gym API over Gymnasium environments.

Supports single and vectorized modes. Normalizes actions, batching, and observation structure.

```mermaid
classDiagram
    class GymnasiumGym {
        + device : torch.device
        + num_envs : int
        + is_vectorized : bool
        + render_mode : str|None
        + observation_space : Space
        + action_space : Space
        --
        + __init__(gym_id: str, vector_env: Env|None, device: str|torch.device, render_mode: str|None, **gym_kwargs)
        + reset(seed: int|None, **reset_kwargs) Observation, dict|list[dict]
        + step(action: torch.Tensor) Observation, float|list[float], bool|list[bool], bool|list[bool], dict|list[dict]
        + sample_action() torch.Tensor
        + close() None
        + render(...) Any
        + get_max_episode_steps() int|None
        + to_observation(raw_obs) Observation
        + vectorize(gym_id: str, num_envs: int, async_mode: bool) GymnasiumGym
    }

    GymnasiumGym --|> Gym
```

## Vector Environments

```python
from physicalai.gyms.gymnasium_gym import GymnasiumGym

# sync vector
env = GymnasiumGym.vectorize("CartPole-v1", num_envs=16)

# async vector
env = GymnasiumGym.vectorize("CartPole-v1", num_envs=8, async_mode=True)
```

## Reset + step

```python
obs, info = env.reset(seed=0)
action = env.sample_action()
obs, reward, terminated, truncated, info = env.step(action)
```
