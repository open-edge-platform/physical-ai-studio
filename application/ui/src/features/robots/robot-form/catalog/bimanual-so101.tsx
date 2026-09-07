import { Flex, Item, Picker, Text } from '@geti-ui/ui';

import { $api } from '../../../../api/client';
import type { SchemaSo101JointCalibration } from '../../../../api/openapi-spec';
import { useProjectId } from '../../../projects/use-project';
import type { SchemaRobot } from '../../robot-types';
import { useRobotForm } from '../provider';

type SourceRobot = Extract<SchemaRobot, { type: 'SO101_Follower' | 'SO101_Leader' }>;

type BimanualSO101Payload = {
    left_serial_number: string;
    right_serial_number: string;
    left_calibration: Record<string, SchemaSo101JointCalibration> | null;
    right_calibration: Record<string, SchemaSo101JointCalibration> | null;
    role: 'follower' | 'leader';
};

const isEligibleSource = (robot: SchemaRobot, type: SourceRobot['type']): robot is SourceRobot =>
    robot.type === type && robot.payload.serial_number !== '' && robot.payload.calibration != null;

interface SO101ArmPickerProps {
    arm: 'left' | 'right';
    robots: SourceRobot[];
    selectedKey: string | number | null;
    onSelect: (robotId: string | number | null) => void;
}

const SO101ArmPicker = ({ arm, robots, selectedKey, onSelect }: SO101ArmPickerProps) => (
    <Picker isRequired label={`Select ${arm} arm`} width='100%' selectedKey={selectedKey} onSelectionChange={onSelect}>
        {robots.map((robot) => (
            <Item key={robot.id} textValue={robot.name}>
                <Text>{robot.name}</Text>
                <Text slot='description'>{robot.payload.serial_number}</Text>
            </Item>
        ))}
    </Picker>
);

export const BimanualSO101FormFields = () => {
    const { project_id } = useProjectId();
    const { activeType, payload, updatePayloadField } = useRobotForm();
    const formPayload = payload as BimanualSO101Payload;
    const robotsQuery = $api.useSuspenseQuery('get', '/api/projects/{project_id}/robots', {
        params: { path: { project_id } },
    });
    const sourceType = activeType === 'BimanualSO101_Leader' ? 'SO101_Leader' : 'SO101_Follower';
    const eligibleRobots = robotsQuery.data.filter((robot) => isEligibleSource(robot, sourceType));
    const leftRobots = eligibleRobots.filter(
        (robot) => robot.payload.serial_number !== formPayload.right_serial_number
    );
    const rightRobots = eligibleRobots.filter(
        (robot) => robot.payload.serial_number !== formPayload.left_serial_number
    );

    const selectArm = (arm: 'left' | 'right', robotId: string | number | null) => {
        const robot = eligibleRobots.find(({ id }) => id === robotId);
        if (robot === undefined || robot.payload.calibration === null) {
            return;
        }
        updatePayloadField(`${arm}_serial_number`, robot.payload.serial_number);
        updatePayloadField(`${arm}_calibration`, robot.payload.calibration);
        updatePayloadField('role', activeType === 'BimanualSO101_Leader' ? 'leader' : 'follower');
    };

    const selectedRobotId = (arm: 'left' | 'right') =>
        eligibleRobots.find((robot) => robot.payload.serial_number === formPayload[`${arm}_serial_number`])?.id ?? null;

    return (
        <Flex direction='column' gap='size-100' width='100%'>
            <SO101ArmPicker
                arm='left'
                robots={leftRobots}
                selectedKey={selectedRobotId('left')}
                onSelect={(robotId) => selectArm('left', robotId)}
            />
            <SO101ArmPicker
                arm='right'
                robots={rightRobots}
                selectedKey={selectedRobotId('right')}
                onSelect={(robotId) => selectArm('right', robotId)}
            />
        </Flex>
    );
};
