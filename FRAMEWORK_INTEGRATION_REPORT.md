# Framework Integration Approaches Report

## Supporting External Frameworks in GetiAction

---

## 🎯 **Executive Summary**

### **The Challenge**

External robotics frameworks like LeRobot have developed a rich ecosystem of state-of-the-art policies (ACT, Diffusion, VQ-BeT, etc.) that we want to leverage in our Lightning-based training pipeline. While we could copy-paste their implementations, this approach presents significant challenges:

- **Maintenance Burden**: Keeping copied code synchronized with upstream improvements and bug fixes
- **Scalability Issues**: Copying becomes unsustainable as the number of policies grows
- **Missing Ecosystem**: Losing access to their data pipelines, evaluation tools, and pretrained models
- **Fragmentation**: Creating divergent implementations that are hard to compare or benchmark

### **Our Solution**

This report outlines **3 distinct approaches** for integrating external robotics frameworks into our Lightning-based training pipeline without copying code. Each approach offers different trade-offs between **ease of implementation**, **performance**, **maintainability**, and **user experience**, allowing us to leverage the full robotics ecosystem while maintaining our Lightning-based architecture.

---

## 📋 **Approach Comparison Matrix**

| Approach | **Implementation Effort** | **Performance** | **User Experience** | **Maintainability** | **Framework Coverage** |
|----------|---------------------------|-----------------|---------------------|---------------------|------------------------|
| **1. Framework-Specific Trainers** | ⭐⭐⭐⭐⭐ Easy | ⭐⭐⭐⭐⭐ Native | ⭐⭐⭐ Multiple APIs | ⭐⭐⭐⭐ Independent | ⭐⭐⭐⭐⭐ Full Access |
| **2. Policy Adapters** | ⭐⭐⭐ Medium | ⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Unified | ⭐⭐⭐ Complex | ⭐⭐⭐ Limited |
| **3. Clean Policy Classes** | ⭐⭐ Hard | ⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Intuitive | ⭐⭐⭐⭐ Modular | ⭐⭐⭐⭐ Good |

---

## 🚀 **Approach 1: Framework-Specific Trainers**

### **Concept**

Create dedicated trainers for each external framework while maintaining a unified interface.

### **Architecture**

```python
# Base trainer interface
class BaseTrainer(FromConfig):
    @abstractmethod
    def fit(self, **kwargs): pass

# Lightning trainer (our native)
class LightningTrainer(BaseTrainer):
    def fit(self, policy: Policy, datamodule: DataModule): pass

# LeRobot trainer (framework-specific)
class LeRobotTrainer(BaseTrainer):
    def fit(self, policy_type: str, dataset_repo: str, **kwargs): pass
```

### **Usage**

```yaml
# Native Lightning training
trainer_type: "lightning"
trainer: {class_path: "LightningTrainer", init_args: {max_epochs: 100}}
model: {class_path: "MyNativePolicy", init_args: {hidden_dim: 512}}

# LeRobot training
trainer_type: "lerobot"
trainer: {class_path: "LeRobotTrainer", init_args: {max_epochs: 100}}
model: {policy_type: "act", hidden_dim: 512}
```

### **Pros & Cons**

✅ **Pros:**

- Fastest implementation
- Native performance (no overhead)
- Full access to framework features
- Easy debugging (direct framework access)
- Leverages our `FromConfig` system

❌ **Cons:**

- Users learn multiple APIs
- Some code duplication across trainers
- Different configuration patterns

---

## 🔄 **Approach 2: Policy Adapters**

### **Concept**

Convert external framework policies to conform to our Lightning Policy interface.

### **Architecture**

```python
class LeRobotPolicyAdapter(Policy):  # our Lightning Policy
    def __init__(self, policy_type: str, hidden_dim: int = 512, ...):
        # Convert explicit args to LeRobot config
        lerobot_config = make_policy_config(policy_type, hidden_dim=hidden_dim, ...)
        self.lerobot_policy = make_policy(lerobot_config)

    def training_step(self, batch, batch_idx):
        # Convert batch format and delegate to LeRobot
        lerobot_batch = self._convert_batch(batch)
        return self.lerobot_policy.compute_loss(lerobot_batch)
```

### **Usage**

```python
# Unified Lightning interface for any framework
trainer = LightningTrainer()
policy = LeRobotPolicyAdapter("act", hidden_dim=512, learning_rate=1e-4)
datamodule = LeRobotDataModule(repo_id="lerobot/dataset")
trainer.fit(policy, datamodule)
```

### **Pros & Cons**

✅ **Pros:**

- Single unified training interface
- Consistent user experience
- Easy framework switching
- Leverages Lightning ecosystem

❌ **Cons:**

- Translation overhead
- Complex format conversions
- May not expose all framework features
- Harder debugging (multiple layers)

---

## 🎨 **Approach 3: Clean Policy Classes**

### **Concept**

Create intuitive policy classes that hide adapter complexity behind clean interfaces.

### **Architecture**

```python
# Internal adapter (hidden from users)
class _LeRobotPolicyAdapter(Policy):
    def __init__(self, policy_type: str, **kwargs):
        # Complex adapter logic

# Clean public interface
class ACT(_LeRobotPolicyAdapter):
    def __init__(self, hidden_dim: int = 512, chunk_size: int = 10,
                 learning_rate: float = 1e-4):
        super().__init__(policy_type="act", hidden_dim=hidden_dim,
                        chunk_size=chunk_size, learning_rate=learning_rate)
```

### **Usage**

```python
# Beautiful, intuitive interface
from getiaction.policies.lerobot import ACT, Diffusion, VQBeT

policy = ACT(hidden_dim=512, chunk_size=10, learning_rate=1e-4)
trainer = LightningTrainer()
trainer.fit(policy, datamodule)
```

### **Pros & Cons**

✅ **Pros:**

- Most intuitive user experience
- Full IDE support (autocomplete, type hints)
- Follows existing patterns (`_LeRobotDatasetAdapter`)
- Easy to discover and use
- Modular design

❌ **Cons:**

- Most complex implementation
- Need to create class for each policy
- Still has adapter overhead

---

## 🗓️ **Recommended Implementation Phases**

### **Phase 1: Foundation (Weeks 1-3)**

**Approach: Framework-Specific Trainers**

**Why start here:**

- Fastest time to value
- Leverages existing `FromConfig` system
- Provides immediate access to external frameworks

**Implementation:**

1. Create `BaseTrainer` interface
2. Implement `LeRobotTrainer` wrapper
3. Add configuration support
4. Test with basic ACT/Diffusion policies

**Deliverables:**

```yaml
# configs/lerobot_act.yaml
trainer_type: "lerobot"
trainer:
  class_path: getiaction.trainers.LeRobotTrainer
  init_args: {max_epochs: 100, learning_rate: 1e-4}
model:
  policy_type: "act"
  hidden_dim: 512
data:
  repo_id: "lerobot/aloha_sim_transfer_cube_human"
```

### **Phase 2: Enhanced Integration (Weeks 4-6)**

**Approach: Clean Policy Classes**

**Why next:**

- Better user experience
- Maintains performance benefits
- Builds on Phase 1 foundation

**Implementation:**

1. Create `_LeRobotPolicyAdapter` base class
2. Implement `ACT`, `Diffusion`, `VQBeT` policy classes
3. Add Lightning training support
4. Create unified configuration system

**Deliverables:**

```python
from getiaction.policies.lerobot import ACT
from getiaction.train import LightningTrainer

policy = ACT(hidden_dim=512, learning_rate=1e-4)
trainer = LightningTrainer()
trainer.fit(policy, datamodule)
```

### **Phase 3: Advanced Features (Weeks 7-8)**

**Approach: Full Policy Adapters (Optional)**

**Why last:**

- Most complex implementation
- Provides unified interface for power users
- Builds on all previous phases

**Implementation:**

1. Create comprehensive policy adapters
2. Add advanced batch conversion logic
3. Implement loss function adapters
4. Add evaluation pipeline adapters

---

## 🎯 **Recommended Strategy: Hybrid Approach**

**Implement ALL approaches in phases** to serve different user needs:

```python
# Phase 1: Quick access (Framework-specific)
lerobot_trainer = LeRobotTrainer()
lerobot_trainer.fit(policy_type="act", dataset_repo="lerobot/dataset")

# Phase 2: Clean interface (Policy classes)
policy = ACT(hidden_dim=512)
lightning_trainer = LightningTrainer()
lightning_trainer.fit(policy, datamodule)

# Phase 3: Power users (Full adapters)
policy = LeRobotPolicyAdapter("act", hidden_dim=512)
trainer.fit(policy, datamodule)  # Full Lightning features
```

---

## 📈 **Success Metrics**

### **Phase 1 Success:**

- [ ] Can train LeRobot ACT policy via config
- [ ] Training performance matches native LeRobot
- [ ] Configuration integrates with existing system

### **Phase 2 Success:**

- [ ] `ACT(hidden_dim=512)` works intuitively
- [ ] Full IDE support (autocomplete, type hints)
- [ ] Works with Lightning trainer seamlessly

### **Phase 3 Success:**

- [ ] Unified interface across all supported frameworks
- [ ] Performance overhead < 5%
- [ ] All Lightning features accessible

---

## 🔚 **Conclusion**

The **hybrid phased approach** provides the best strategy:

1. **Immediate value** with framework-specific trainers
2. **Better UX** with clean policy classes
3. **Advanced features** with full adapters

This strategy ensures **continuous delivery of value** while building toward a comprehensive framework integration solution.

---

## 📝 **Implementation Notes**

### **Key Design Principles**

1. **Leverage existing patterns**: Use our `FromConfig` mixin and `_LeRobotDatasetAdapter` patterns
2. **Gradual complexity**: Start simple, add sophistication over time
3. **User choice**: Support multiple approaches for different use cases
4. **Performance conscious**: Minimize overhead in critical paths
5. **Maintainable**: Keep each approach independently maintainable

### **Technical Considerations**

- **Data format conversion**: Critical for adapter approaches
- **Configuration management**: Ensure consistent interface across approaches
- **Error handling**: Provide clear error messages for each approach
- **Testing strategy**: Each approach needs comprehensive testing
- **Documentation**: Clear examples for each approach and use case

### **Risk Mitigation**

- **Start with framework-specific trainers** to reduce implementation risk
- **Validate performance** at each phase before proceeding
- **Maintain backward compatibility** as you add new approaches
- **Plan for framework evolution** (LeRobot API changes, new frameworks)
