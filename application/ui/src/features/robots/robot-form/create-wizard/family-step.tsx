import { Flex, Grid, minmax, repeat } from '@geti-ui/ui';

import { CardMeta, CardTitle, RobotCard } from './robot-card';
import type { RobotFamily } from './robot-families';
import { RoleBadge } from './role-badge';
import { StepHeader } from './step-header';

interface FamilyStepProps {
    families: RobotFamily[];
    selectedKey: string | null;
    onSelect: (family: RobotFamily) => void;
}

export const FamilyStep = ({ families, selectedKey, onSelect }: FamilyStepProps) => (
    <>
        <StepHeader
            title='Which robot are you adding?'
            description="Pick the hardware. You'll choose follower/leader next."
        />
        <Grid columns={repeat('auto-fill', minmax('size-3400', '1fr'))} gap='size-200'>
            {families.map((family) => (
                <RobotCard
                    key={family.key}
                    type={family.variants[0].type}
                    thumbnailHeight='size-2000'
                    isSelected={family.key === selectedKey}
                    onSelect={() => onSelect(family)}
                >
                    <CardTitle>{family.label}</CardTitle>
                    <Flex alignItems='center' justifyContent='space-between' gap='size-75'>
                        <CardMeta>
                            {family.variants.length} variant{family.variants.length === 1 ? '' : 's'}
                        </CardMeta>
                        <Flex gap='size-50'>
                            {[...new Set(family.variants.map(({ role }) => role))].map((role) => (
                                <RoleBadge key={role} role={role} />
                            ))}
                        </Flex>
                    </Flex>
                </RobotCard>
            ))}
        </Grid>
    </>
);
