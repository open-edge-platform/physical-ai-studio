# Advanced Patterns

Add these after the minimal policy works. Keep each feature in its owning mixin.

## PEFT and LoRA

Use all three PEFT layers:

```python
class MyModelConfig(PeftConfigMixin, Config):
    ...

class MyModel(PeftModelMixin, TemplateModel):
    @classmethod
    def get_default_peft_targets(cls) -> tuple[str, ...]:
        return ("action_head",)

class MyPolicy(PeftPolicyMixin, TemplatePolicy):
    ...
```

Order:

1. Create the base model.
2. Load pretrained base weights.
3. Inject adapters.
4. Load checkpoint tensors.

PEFT settings belong in the saved config. Default target modules belong on the model.

## Gradient Checkpointing

Keep enablement on the model. The policy only invokes it.

```python
if self.gradient_checkpointing:
    self.model.gradient_checkpointing_enable()
```

Apply it after model creation and base-weight loading.

## Real-Time Chunking

Use `RTCPolicyMixin` and `RTCModelMixin`. Sync state after model creation.

RTC must receive the full action chunk. Trim to `n_action_steps` afterward.

## Checkpoint Restoration

The shared policy base owns the normal hooks:

```python
def on_save_checkpoint(self, checkpoint):
    checkpoint["model_config"] = self.config.to_dict()

@classmethod
def load_from_checkpoint(cls, path, **kwargs):
    kwargs["pretrained_name_or_path"] = None
    return super().load_from_checkpoint(path, **kwargs)

def on_load_checkpoint(self, checkpoint):
    self._config = MyModelConfig.from_dict(checkpoint["model_config"])
    self.configure_model()
```

Clear the pretrained path before construction. Build the final module structure before
Lightning loads tensors. Extend these hooks only for policy-specific state.

## Feature Adaptation

Allow feature changes only when model architecture stays valid.

1. Validate the action width.
2. Replace the config feature lists.
3. Rebuild processors.
4. Reset runtime queues.
5. Do not rebuild the model.

Renaming must preserve metadata and order.

## Export Exceptions

Keep export code in a dedicated policy export mixin.

Use schemas, samples, backend parameters, and supported backends first. Override
`_get_default_export_input_sample()` only when those hooks cannot prepare tracing
inputs. Override a backend method only as a last resort.

```python
class MyPolicyExportMixin(ExportablePolicyMixin):
    def _get_default_export_input_sample(self):
        sample = super()._get_default_export_input_sample()
        return None if sample is None else adapt_trace_inputs(sample)
```

Private-hook overrides need focused export tests.

See [Policy Migration](migration.md) for migration order.
