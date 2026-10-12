import { describe, expect, it } from 'vitest';

import { getEstimatedTrainingSteps } from './training-steps';

describe('getEstimatedTrainingSteps', () => {
    it('estimates steps from the expected training samples', () => {
        expect(getEstimatedTrainingSteps([100, 100], 8, 5)).toBe(60);
        expect(getEstimatedTrainingSteps([10, 100], 8, 5)).toBe(30);
    });
});
