# PushTGym

Convenience wrapper around Gymnasium PushT. Provides typed image+position observations.

```mermaid
classDiagram
    class PushTGym {
        + __init__(gym_id: str, obs_type: str, device: torch.device)
        + convert_raw_to_observation(raw_obs, camera_keys) Observation
    }

    PushTGym --|> GymnasiumGym
```

Example:

```python
from physicalai.gyms import PushTGym

# default configuration
env = PushTGym()

obs, info = env.reset(seed=0)
action = env.sample_action()

obs, reward, terminated, truncated, info = env.step(action)
```
