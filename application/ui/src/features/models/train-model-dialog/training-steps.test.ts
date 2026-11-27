import { describe, expect, it } from '@rstest/core';

import { getEstimatedTrainingSteps } from './training-steps';

describe('getEstimatedTrainingSteps', () => {
    it('estimates steps from the expected training samples', () => {
        expect(getEstimatedTrainingSteps([100, 100], 8, 5)).toBe(60);
        expect(getEstimatedTrainingSteps([10, 100], 8, 5)).toBe(30);
    });

    it('uses the selected recipe validation split, including disabled validation', () => {
        expect(getEstimatedTrainingSteps([100, 100, 100, 100], 8, 5, 0.5)).toBe(125);
        expect(getEstimatedTrainingSteps([100], 8, 5, 0)).toBe(60);
    });
});
