import { describe, expect, it } from '@rstest/core';

import { getMockedRobotCatalogDefinition } from '../../../../test-utils/mocks/mock-robot-catalog-definition';
import { getReachableSteps } from './get-reachable-steps';
import type { RobotFamily } from './robot-families';

const family: RobotFamily = {
    key: 'SO101',
    label: 'SO101',
    variants: [
        getMockedRobotCatalogDefinition({ type: 'SO101_Follower' }),
        getMockedRobotCatalogDefinition({ type: 'SO101_Leader', role: 'leader' }),
    ],
};

describe('getReachableSteps', () => {
    it('only allows choosing a robot when no family is chosen', () => {
        expect(getReachableSteps(undefined, undefined)).toEqual(['robot']);
    });

    it('allows the variant step, but not configure, once a family is chosen', () => {
        expect(getReachableSteps(family, undefined)).toEqual(['robot', 'variant']);
    });

    it('does not allow configure for a type that belongs to another family', () => {
        expect(getReachableSteps(family, 'Other_Follower')).toEqual(['robot', 'variant']);
    });

    it('allows every step once a variant of the family is chosen', () => {
        expect(getReachableSteps(family, 'SO101_Leader')).toEqual(['robot', 'variant', 'configure']);
    });

    it('keeps the variant step reachable for single-variant families', () => {
        const single: RobotFamily = { key: 'Solo', label: 'Solo', variants: [family.variants[0]] };

        expect(getReachableSteps(single, undefined)).toEqual(['robot', 'variant']);
    });
});
