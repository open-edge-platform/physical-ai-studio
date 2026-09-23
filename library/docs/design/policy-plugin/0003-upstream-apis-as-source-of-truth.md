---
status: accepted
---

# Treat upstream APIs and dataset contracts as the source of truth

Upstream integrations use the upstream project's own classes and dataset formats rather than Studio copies of them. Pass a live official object when its class is importable in the caller's Python environment; otherwise pass a `class_path` and `init_args` reference to the same class, which the plugin reconstructs in the upstream Python environment. Studio-owned mirror classes would duplicate upstream schemas, drift with upstream releases, and add work for every contributor.

## Consequences

- The class path of the official object selects the plugin and defines the classes a worker may instantiate.
- Studio adds only a small envelope, such as `model`, `dataset`, `output_dir`, and `python_env`. Each value binds to its official field; supplying both an envelope value and the corresponding official field is an error.
- Plugins supply dataset converters as library code shared by Python, CLI, and GUI. Conversion from Studio's datasets to official formats comes first; conversion into Studio follows where the mapping is well defined.
- Clear-cut mappings convert automatically. Mappings requiring a semantic choice use a reviewed mapping, and impossible mappings fail with a clear reason. Originals are preserved, derived datasets are cached with provenance, and lossless round trips are not promised.
- Convert Studio's LeRobot datasets to GR00T's format with library code running in Studio's Python environment, and verify the output with GR00T's own dataset loader. This deliberately does not reuse NVIDIA's conversion script, which requires Python earlier than 3.12 and therefore a separate environment. Offer upstream a converter that supports current Python.
- Keep dataset conversion distinct from `FormatConverter`, which converts batches in memory. The general dataset-conversion interface belongs in core `physicalai.data`; format-specific converters belong to plugins.
- Split each upstream plugin into a lightweight part installed in Studio's environment, covering discovery, configuration binding, and converters, and a worker part that runs in the upstream Python environment.
- Live objects do not cross into Python environments with different dependencies. Mutated in-memory models require an explicit artifact or a compatible process.
- First-party policies follow the same rule: Studio's classes are their official API.

See the [experience proposal](policy-integration-experience.md) for examples.
