import { useMemo } from 'react';

import type { SchemaRobotCatalogDefinitionResponse } from '../../../../api/openapi-spec';
import { useRobotCatalogQuery } from '../../robot-catalog.hooks';

export type CatalogEntry = SchemaRobotCatalogDefinitionResponse;

export interface RobotFamily {
    key: string;
    label: string;
    variants: CatalogEntry[];
}

// The catalog has no family/variant fields, so families are derived from the type name:
// <Family>[_<HW>]_<Role>, where HW is an optional two-capital-letter hardware code.
// SO101_Follower -> SO101, ReBot_B601_RS_Follower -> ReBot_B601, StarArm_102_LD_Leader -> StarArm_102
const withoutRole = (type: string) => type.replace(/_(Follower|Leader)$/, '');
const withoutHardware = (type: string) => type.replace(/_[A-Z]{2}$/, '');

export const getFamilyKey = (type: string) => withoutHardware(withoutRole(type));

const stripRoleWords = (label: string) => label.replace(/\b(Follower|Leader)\b/gi, '').trim();

export const groupCatalogIntoFamilies = (catalog: CatalogEntry[]): RobotFamily[] => {
    const byKey = new Map<string, CatalogEntry[]>();
    catalog.forEach((entry) => {
        const key = getFamilyKey(entry.type);
        byKey.set(key, [...(byKey.get(key) ?? []), entry]);
    });

    return Array.from(byKey.entries()).map(([key, variants]) => {
        const sorted = variants.toSorted((a, b) =>
            a.role === b.role ? a.display_name.localeCompare(b.display_name) : a.role === 'follower' ? -1 : 1
        );
        return { key, label: stripRoleWords(sorted[0].display_name) || key.replaceAll('_', ' '), variants: sorted };
    });
};

export const useRobotFamilies = (): RobotFamily[] => {
    const { data: catalog } = useRobotCatalogQuery();

    return useMemo(() => groupCatalogIntoFamilies(catalog), [catalog]);
};
