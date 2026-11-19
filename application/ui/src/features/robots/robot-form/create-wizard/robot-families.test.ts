import { describe, expect, it } from '@rstest/core';

import { getMockedRobotCatalogDefinition } from '../../../../test-utils/mocks/mock-robot-catalog-definition';
import { getFamilyKey, groupCatalogIntoFamilies } from './robot-families';

describe('getFamilyKey', () => {
    it.each([
        ['SO101_Follower', 'SO101'],
        ['SO101_Leader', 'SO101'],
    ])('drops the role suffix from %s', (type, expected) => {
        expect(getFamilyKey(type)).toBe(expected);
    });

    it.each([
        ['ReBot_B601_DM_Follower', 'ReBot_B601'],
        ['ReBot_B601_RS_Follower', 'ReBot_B601'],
        ['StarArm_102_LD_Leader', 'StarArm_102'],
        ['StarArm_102_HD_Leader', 'StarArm_102'],
        ['StarArm_102_FL_Follower', 'StarArm_102'],
    ])('drops the hardware code and the role from %s', (type, expected) => {
        expect(getFamilyKey(type)).toBe(expected);
    });

    it.each(['Trossen_WidowXAI_Follower', 'Trossen_Bimanual_WidowXAI_Leader', 'BimanualSO101_Follower'])(
        'does not mistake part of the name of %s for a hardware code',
        (type) => {
            expect(getFamilyKey(type)).toBe(type.replace(/_(Follower|Leader)$/, ''));
        }
    );

    it('leaves a type without a role suffix unchanged', () => {
        expect(getFamilyKey('Solo_Arm')).toBe('Solo_Arm');
    });

    it('only drops the role when it is the trailing segment', () => {
        expect(getFamilyKey('Leader_Arm_Follower')).toBe('Leader_Arm');
        expect(getFamilyKey('Follower_Arm')).toBe('Follower_Arm');
    });
});

describe('groupCatalogIntoFamilies', () => {
    it('returns no families for an empty catalog', () => {
        expect(groupCatalogIntoFamilies([])).toEqual([]);
    });

    it('groups a follower and its leader into one family', () => {
        const follower = getMockedRobotCatalogDefinition({ type: 'SO101_Follower', display_name: 'SO101 Follower' });
        const leader = getMockedRobotCatalogDefinition({
            type: 'SO101_Leader',
            display_name: 'SO101 Leader',
            role: 'leader',
        });

        const families = groupCatalogIntoFamilies([follower, leader]);

        expect(families).toHaveLength(1);
        expect(families[0].key).toBe('SO101');
        expect(families[0].variants).toEqual([follower, leader]);
    });

    it('lists followers before leaders whatever the catalog order', () => {
        const follower = getMockedRobotCatalogDefinition({ type: 'SO101_Follower', display_name: 'SO101 Follower' });
        const leader = getMockedRobotCatalogDefinition({
            type: 'SO101_Leader',
            display_name: 'SO101 Leader',
            role: 'leader',
        });

        const [family] = groupCatalogIntoFamilies([leader, follower]);

        expect(family.variants.map(({ role }) => role)).toEqual(['follower', 'leader']);
    });

    it('groups the DM and RS hardware variants of the ReBot B601 into one family', () => {
        const families = groupCatalogIntoFamilies([
            getMockedRobotCatalogDefinition({ type: 'ReBot_B601_DM_Follower', display_name: 'ReBot B601 DM Follower' }),
            getMockedRobotCatalogDefinition({ type: 'ReBot_B601_RS_Follower', display_name: 'ReBot B601 RS Follower' }),
        ]);

        expect(families).toHaveLength(1);
        expect(families[0].key).toBe('ReBot_B601');
        expect(families[0].variants.map(({ type }) => type)).toEqual([
            'ReBot_B601_DM_Follower',
            'ReBot_B601_RS_Follower',
        ]);
    });

    it('groups the Star Arm 102 follower with both of its leaders, follower first', () => {
        const families = groupCatalogIntoFamilies([
            getMockedRobotCatalogDefinition({
                type: 'StarArm_102_LD_Leader',
                display_name: 'Star Arm 102-LD Leader',
                role: 'leader',
            }),
            getMockedRobotCatalogDefinition({
                type: 'StarArm_102_HD_Leader',
                display_name: 'Star Arm 102-HD Leader',
                role: 'leader',
            }),
            getMockedRobotCatalogDefinition({
                type: 'StarArm_102_FL_Follower',
                display_name: 'Star Arm 102-FL Follower',
            }),
        ]);

        expect(families).toHaveLength(1);
        expect(families[0].variants.map(({ type }) => type)).toEqual([
            'StarArm_102_FL_Follower',
            'StarArm_102_HD_Leader',
            'StarArm_102_LD_Leader',
        ]);
    });

    it('keeps the order in which families first appear in the catalog', () => {
        const families = groupCatalogIntoFamilies([
            getMockedRobotCatalogDefinition({ type: 'B_Follower', display_name: 'B Follower' }),
            getMockedRobotCatalogDefinition({ type: 'A_Follower', display_name: 'A Follower' }),
            getMockedRobotCatalogDefinition({ type: 'B_Leader', display_name: 'B Leader', role: 'leader' }),
        ]);

        expect(families.map(({ key }) => key)).toEqual(['B', 'A']);
    });

    it('labels a family with the display name of its follower without the role word', () => {
        const [family] = groupCatalogIntoFamilies([
            getMockedRobotCatalogDefinition({
                type: 'SO101_Leader',
                display_name: 'SO-101 Leader',
                role: 'leader',
            }),
            getMockedRobotCatalogDefinition({ type: 'SO101_Follower', display_name: 'SO-101 Follower' }),
        ]);

        expect(family.label).toBe('SO-101');
    });

    it('labels a leader-only family from its leader display name', () => {
        const [family] = groupCatalogIntoFamilies([
            getMockedRobotCatalogDefinition({
                type: 'Master_Leader',
                display_name: 'Master Arm Leader',
                role: 'leader',
            }),
        ]);

        expect(family.label).toBe('Master Arm');
    });

    it('falls back to the readable family key when the display name is only a role word', () => {
        const [family] = groupCatalogIntoFamilies([
            getMockedRobotCatalogDefinition({ type: 'Solo_Arm_Follower', display_name: 'Follower' }),
        ]);

        expect(family.label).toBe('Solo Arm');
    });
});
