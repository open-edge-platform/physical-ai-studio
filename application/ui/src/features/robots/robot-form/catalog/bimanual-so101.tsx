import { Flex, Item, Picker, Text } from '@geti-ui/ui';

import { $api } from '../../../../api/client';
import { useProjectId } from '../../../projects/use-project';
import type { SchemaRobot } from '../../robot-types';
import { useRobotForm } from '../provider';

type SourceRobot = Extract<SchemaRobot, { type: 'SO101_Follower' | 'SO101_Leader' }>;

const isEligibleSource = (robot: SchemaRobot, type: SourceRobot['type']): robot is SourceRobot =>
    robot.type === type && robot.payload.serial_number !== '' && robot.payload.calibration != null;

// TODO: can we replace this using recursive form fields?
// make the left and right payload be a so101 payload
export const BimanualSO101FormFields = () => {
    const { project_id } = useProjectId();
    const { activeType, payload, updatePayloadField } = useRobotForm();
    const robots = $api.useSuspenseQuery('get', '/api/projects/{project_id}/robots', {
        params: { path: { project_id } },
    });
    const sourceType = activeType === 'BimanualSO101_Leader' ? 'SO101_Leader' : 'SO101_Follower';
    const eligible = robots.data.filter((robot) => isEligibleSource(robot, sourceType));
    const selectArm = (arm: 'left' | 'right', robotId: string | number | null) => {
        const robot = eligible.find(({ id }) => id === robotId);
        if (robot === undefined || robot.payload.calibration === null) return;
        updatePayloadField(`${arm}_serial_number`, robot.payload.serial_number);
        updatePayloadField(`${arm}_calibration`, robot.payload.calibration);
        updatePayloadField('role', activeType === 'BimanualSO101_Leader' ? 'leader' : 'follower');
    };
    const picker = (arm: 'left' | 'right') => {
        const other = payload[`${arm === 'left' ? 'right' : 'left'}_serial_number`];
        const selected =
            eligible.find((robot) => robot.payload.serial_number === payload[`${arm}_serial_number`])?.id ?? null;
        return (
            <Picker
                isRequired
                label={`Select ${arm} arm`}
                width='100%'
                selectedKey={selected}
                onSelectionChange={(id) => selectArm(arm, id)}
            >
                {eligible
                    .filter((robot) => robot.payload.serial_number !== other)
                    .map((robot) => (
                        <Item key={robot.id} textValue={robot.name}>
                            <Text>{robot.name}</Text>
                            <Text slot='description'>{robot.payload.serial_number}</Text>
                        </Item>
                    ))}
            </Picker>
        );
    };

    return (
        <Flex direction='column' gap='size-100' width='100%'>
            {picker('left')}
            {picker('right')}
        </Flex>
    );
};
