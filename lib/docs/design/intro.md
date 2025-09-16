# Geti action trainer design

Welcome to the `action_trainer` package.

```mermaid
graph TD
    A["action_trainer"]
    A --> B["data/ <br/> Dataset management"]
    A --> C["gyms/ <br/> Simulated gym environments"]
    A --> D["policy/ <br/> Policies"]
    A --> E["train/ <br/> Trainers and Metrics]
```
