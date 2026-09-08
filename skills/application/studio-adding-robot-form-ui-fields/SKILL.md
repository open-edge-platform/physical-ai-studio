---
name: studio-adding-robot-form-ui-fields
description: Adds a new interactive robot form UI field for plugin payload schemas. Use when introducing a new `robot_payload_ui` item kind, wiring renderer support under application/ui/src/features/robots/robot-form/robot-schema/components, updating plugin SDK UI-schema validation, and documenting how plugin authors adopt the field.
license: Apache-2.0
---

# Adding Robot Form UI Field Components

Add new robot-form controls as explicit `robot_payload_ui(...)` item kinds so plugin authors can opt in without Studio-specific React code in plugins.

Read these first:

- `application/docs/robot-plugins.md`
- `application/docs/explanation/robot-plugin-architecture.md`
- `application/plugin/src/physicalai_studio_plugin/ui_schema.py`
- `application/ui/src/features/robots/robot-form/robot-schema/schema-form.tsx`

## Workflow

1. **Define the UX contract and ownership model.**

   - Pick an explicit `kind` name (for example `calibration`) and required item shape (`name`, optional `label`, optional `description`, etc.).
   - Keep ownership semantics aligned with existing controls: item-owned fields must not also be rendered as `field` or owned by another item.
   - Prefer explicit opt-in (`kind`) over heuristic detection in the UI.
   - Done when: you can describe exactly which payload field(s) the item owns and how plugins enable it.

2. **Add SDK typing + validation for the new item kind.**

   - Edit `application/plugin/src/physicalai_studio_plugin/ui_schema.py`:
     - add a new `TypedDict` for the item kind,
     - include it in `RobotUiItem` union,
     - validate required fields and field type constraints,
     - enforce ownership conflict checks.
   - Keep validation errors precise and actionable for plugin authors.
   - Done when: invalid metadata fails `validate_robot_payload_ui(...)` with a clear error and valid metadata passes.

3. **Render the field in SchemaForm using kind-based dispatch.**

   - Add a component under `application/ui/src/features/robots/robot-form/robot-schema/components/`.
   - Integrate it in `application/ui/src/features/robots/robot-form/robot-schema/schema-form.tsx` in `SchemaFormItem` next to existing `connection` / `ip_address` handling.
   - Reuse shared visibility rules; do not create ad-hoc visibility behavior for a single kind.
   - Honor `robot_field_ui({"advanced_configuration": True})` and required/optional behavior consistently.
   - Done when: rendering is triggered only by explicit `kind` items and matches default form behavior for advanced/required fields.

4. **Implement robust field UX and payload updates.**

   - Parse and validate user input in the component before mutating payload.
   - Show clear inline errors for invalid input.
   - Keep labels/descriptions visually consistent with existing form controls.
   - For structured payloads (like calibration maps), include a compact preview so users can verify imported values.
   - Done when: happy path updates payload correctly, invalid path is recoverable, and the control is readable in dense forms.

5. **Adopt the new kind in built-in catalog payloads.**

   - Update backend payload UI metadata in the relevant catalog file(s), e.g. `application/backend/src/robots/catalog/so101.py`.
   - Replace generic `field` usage with the new item kind where appropriate.
   - Keep business semantics in field descriptions (for example: when a provided calibration bypasses guided calibration).
   - Done when: `/api/robots/catalog/{type}/schema` emits the expected `x-physicalai-ui` item and Studio renders the new control.

6. **Add tests at all affected layers.**

   - UI component tests (new file):
     - `application/ui/src/features/robots/robot-form/robot-schema/components/<new-field>.test.tsx`
     - cover parse success, parse failure, required/optional markers, preview/sorting, and clear/reset behavior.
   - Schema form integration tests:
     - `application/ui/src/features/robots/robot-form/robot-schema/schema-form.test.tsx`
     - assert kind-driven rendering, advanced visibility, and payload wiring.
   - Feature-level tests where used (example bimanual forms):
     - `application/ui/src/features/robots/robot-form/catalog/*.test.tsx`
   - Plugin SDK contract tests:
     - `application/plugin/tests/test_contracts.py`
     - cover valid item metadata, type errors, missing fields, and ownership conflicts.
   - Done when: all changed test suites pass locally.

7. **Update docs and communicate to plugin authors.**

   - Update public docs:
     - `application/docs/robot-plugins.md` (supported item kinds + usage snippet)
     - `application/docs/explanation/robot-plugin-architecture.md` (architecture list of supported kinds)
   - Add a handoff note for plugin maintainers when behavior changes materially:
     - `application/docs/handoff-<feature>.md` (expected JSON format, migration guidance, limitations)
   - Include copy-paste plugin snippet showing new `robot_payload_ui` usage.
   - Call out rollout notes explicitly for affected plugin owners (for example SO101, BimanualSO101, LeKiwi):
     - what to change,
     - what stays backward compatible,
     - what validation/runtime behavior changes.
   - Done when: plugin authors can adopt the feature without reading UI source code.

## Verify

From `application/ui/`:

```bash
npm run type-check
npm run test:unit -- src/features/robots/robot-form/robot-schema/components/<new-field>.test.tsx
npm run test:unit -- src/features/robots/robot-form/robot-schema/schema-form.test.tsx
```

From repo root (or environment where plugin tests run):

```bash
uv run python -m pytest application/plugin/tests/test_contracts.py
```

When skill files changed:

```bash
python3 .github/scripts/skills/agent_skills.py sync
python3 .github/scripts/skills/agent_skills.py validate
```

## References

- `application/ui/src/features/robots/robot-form/robot-schema/schema-form.tsx`
- `application/ui/src/features/robots/robot-form/robot-schema/components/connection-field.tsx`
- `application/ui/src/features/robots/robot-form/robot-schema/components/ip-address-field.tsx`
- `application/ui/src/features/robots/robot-form/robot-schema/components/calibration-field.tsx`
- `application/plugin/src/physicalai_studio_plugin/ui_schema.py`
- `application/plugin/tests/test_contracts.py`
