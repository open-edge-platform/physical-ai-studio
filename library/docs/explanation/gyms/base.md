# Gym

Abstract interface for unified environment interaction.

Gym defines the backend-agnostic API used across action environments.
All environments must implement reset, step, sampling, and observation conversion.

```mermaid
classDiagram
    class Gym {
        <<abstract>>
        + reset(seed: int | None, **reset_kwargs) Observation, dict|list[dict]
        + step(action: torch.Tensor) Observation, float|list[float], bool|list[bool], bool|list[bool], dict|list[dict]
        + render(*args, **kwargs) Any
        + close() None
        + sample_action() torch.Tensor
        + to_observation(raw_obs: Any) Observation
    }

```
