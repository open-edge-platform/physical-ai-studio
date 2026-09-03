/**
 * SnapFlow distillation compresses a flow-matching policy's multi-step
 * denoising loop into a single forward pass, which is where most of a VLA's
 * inference latency sits. It only applies to flow-matching policies, so the
 * training dialog hides the option for everything else.
 *
 * Mirrors `SNAPFLOW_POLICIES` in `application/backend/src/training/job.py`;
 * the backend rejects a mismatch, this only keeps the UI from offering one.
 */
const SNAPFLOW_POLICIES: ReadonlySet<string> = new Set(['pi05', 'smolvla']);

/** Whether a policy can be distilled with SnapFlow. */
export const supportsSnapflow = (policy: string): boolean => SNAPFLOW_POLICIES.has(policy.toLowerCase());

/** Label shown on the badge marking a distilled model or a distilling job. */
export const SNAPFLOW_BADGE_TEXT = 'SnapFlow';

/** Tooltip explaining what the badge means, shared by the job and model rows. */
export const SNAPFLOW_BADGE_TITLE = 'Distilled with SnapFlow: generates an action chunk in a single denoising step';

/**
 * Badge colour. Distinct from the status badges (energy blue / red) and the
 * remote-trainer badge (purple), so the three never read as one another.
 */
export const SNAPFLOW_BADGE_COLOR = 'var(--brand-rust)';
