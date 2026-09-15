/**
 * LoRA/DoRA fine-tuning freezes the base model and trains small low-rank
 * adapters instead, cutting memory use and training time. It only applies to
 * policies whose config mixes in `PeftConfigMixin`, so the training dialog
 * hides the option for everything else.
 *
 * Mirrors `PEFT_POLICIES` in `application/backend/src/training/job.py`; the
 * backend rejects a mismatch, this only keeps the UI from offering one.
 */
const PEFT_POLICIES: ReadonlySet<string> = new Set(['pi05', 'pi0']);

/** Whether a policy can be fine-tuned with LoRA/DoRA. */
export const supportsLora = (policy: string): boolean => PEFT_POLICIES.has(policy.toLowerCase());

/** Label shown on the badge marking a LoRA fine-tuned job or model. */
export const LORA_BADGE_TEXT = 'LoRA';

/** Label shown instead of `LORA_BADGE_TEXT` when the DoRA variant was used. */
export const DORA_BADGE_TEXT = 'DoRA';

/** Tooltip explaining what the LoRA badge means. */
export const LORA_BADGE_TITLE = 'Fine-tuned with LoRA: adapts the base model via small trainable low-rank adapters';

/** Tooltip explaining what the DoRA badge means. */
export const DORA_BADGE_TITLE = 'Fine-tuned with DoRA: LoRA adaptation plus a learned per-column magnitude vector';

/**
 * Badge colour. Distinct from the status badges (energy blue / red) and the
 * remote-trainer badge (purple), so the badges never read as one another.
 */
export const LORA_BADGE_COLOR = '#7b61ff';
