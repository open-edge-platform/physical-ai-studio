import { Grid, minmax, repeat } from '@geti-ui/ui';

import { useRobotForm } from '../provider';
import { CardMeta, CardTitle, RobotCard } from './robot-card';
import type { RobotFamily } from './robot-families';
import { StepHeader } from './step-header';

interface VariantStepProps {
    family: RobotFamily;
    onSelect: (type: string) => void;
}

export const VariantStep = ({ family, onSelect }: VariantStepProps) => {
    const { activeType } = useRobotForm();

    return (
        <>
            <StepHeader
                title={`${family.label}: which variant?`}
                description='Followers execute policies; leaders are used for teleoperation.'
            />
            <Grid columns={repeat('auto-fill', minmax('size-3400', '1fr'))} gap='size-200'>
                {family.variants.map((entry) => (
                    <RobotCard
                        key={entry.type}
                        type={entry.type}
                        thumbnailHeight='size-2000'
                        isSelected={entry.type === activeType}
                        onSelect={() => onSelect(entry.type)}
                    >
                        <CardTitle>{entry.display_name}</CardTitle>
                        {entry.zero_calibration != null && <CardMeta>Requires zero-pose calibration</CardMeta>}
                    </RobotCard>
                ))}
            </Grid>
        </>
    );
};
