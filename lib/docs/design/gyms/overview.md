# Gyms

Gym environments play a crucial role in evaluating using simulation data.

```mermaid
graph TD
    A["action_trainer"]
    A --> B["gyms/ <br/> Gyms"]
    B --> C["__init__.py"]
    B --> D["base.py"]
    B --> E["pusht.py"]
```

This section describes the design for the action_trainer.data module.

```bash
action_trainer
├── gyms/  # Gyms
│   ├── __init__.py
│   ├── base.py
│   ├── pusht.py
```
