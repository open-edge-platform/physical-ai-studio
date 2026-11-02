import { requiresFeatureMapping } from './policy-camera-slots';

const ALL_WIZARD_STEPS = ['setup', 'feature-mapping', 'training-parameters', 'export'] as const;

export type WizardStep = (typeof ALL_WIZARD_STEPS)[number];

/**
 * The steps to walk for a policy.
 *
 * Only a policy pretrained on a fixed camera order has anything to map, so for
 * every other policy the feature-mapping step is dropped rather than shown empty.
 * Retraining drops it too: a resumed policy keeps the camera layout stored in its
 * checkpoint, so a mapping picked here would be ignored.
 */
export const getWizardSteps = (policy: string, isRetraining = false): WizardStep[] =>
    ALL_WIZARD_STEPS.filter((step) => step !== 'feature-mapping' || (requiresFeatureMapping(policy) && !isRetraining));
