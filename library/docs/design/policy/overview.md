# Policies

In action training framework, polices represent action models.
Each policy consists of a Lightning module and actual torch model.
Torch model ideally should depend only on torch entities,
and be easilly extractible from the framework.

```mermaid
classDiagram
    TrainerModule
    TrainerModule : ActionModel

    Model
    Model: +forward(dict[str, Tensor]) -> Tensor
    Model: +select_action(dict[str, Tensor]) -> Tensor
    Model: +predict_action_chunk(dict[str, Tensor]) -> Tensor
```

We will lay out each new policy with this tree structure:

```bash
├──policy_name
│   ├── config.py
│   ├── model.py
│   └── policy.py
```
