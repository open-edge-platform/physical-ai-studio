import { Flex, Item, Picker, Text } from '@geti-ui/ui';

import { $api } from '../../../../api/client';
import type { SchemaSo101JointCalibration } from '../../../../api/openapi-spec';
import { useProjectId } from '../../../projects/use-project';
import type { ConfigurableRobotType, SchemaRobot, SchemaRobotInput, SchemaRobotType } from '../../robot-types';
import { useRobotFormFields } from '../provider';

// Bimanual SO101 is a plugin robot type; the backend does not return it in the OpenAPI catalog, so its payload schema
// is defined here.
type BimanualSO101Payload = {
    left_serial_number: string;
    right_serial_number: string;
    left_calibration: Record<string, SchemaSo101JointCalibration> | null;
    right_calibration: Record<string, SchemaSo101JointCalibration> | null;
    baudrate: number;
    role: 'follower' | 'leader';
    disable_torque_on_disconnect: boolean;
};

export interface BimanualSO101FormData {
    name: string;
    payload: BimanualSO101Payload;
}

type SO101SourceRobot = Extract<SchemaRobot, { type: 'SO101_Follower' | 'SO101_Leader' }>;
type BimanualSO101Robot = SchemaRobot & {
    type: 'BimanualSO101_Follower' | 'BimanualSO101_Leader';
    payload: BimanualSO101Payload;
};

const isBimanualSO101Robot = (robot: SchemaRobot): robot is BimanualSO101Robot =>
    robot.type === 'BimanualSO101_Follower' || robot.type === 'BimanualSO101_Leader';

export const getInitialBimanualSO101FormData = (robot?: SchemaRobot): BimanualSO101FormData => ({
    name: robot?.name ?? '',
    payload:
        robot && isBimanualSO101Robot(robot)
            ? robot.payload
            : {
                  left_serial_number: '',
                  right_serial_number: '',
                  left_calibration: null,
                  right_calibration: null,
                  baudrate: 1000000,
                  role: 'follower',
                  disable_torque_on_disconnect: true,
              },
});

export const buildBimanualSO101Body = (
    formData: BimanualSO101FormData,
    schemaType: SchemaRobotType,
    robot_id: string
): SchemaRobotInput | null => {
    if (
        !formData.payload.left_serial_number ||
        !formData.payload.right_serial_number ||
        !formData.payload.left_calibration ||
        !formData.payload.right_calibration
    ) {
        return null;
    }

    return {
        id: robot_id,
        name: formData.name,
        type: schemaType,
        payload: {
            ...formData.payload,
            role: schemaType === 'BimanualSO101_Leader' ? 'leader' : 'follower',
        },
    } as unknown as SchemaRobotInput;
};

const isEligibleSource = (robot: SchemaRobot, type: SO101SourceRobot['type']): robot is SO101SourceRobot =>
    robot.type === type && robot.payload.serial_number !== '' && robot.payload.calibration != null;

interface SO101ArmPickerProps {
    arm: 'left' | 'right';
    robots: SO101SourceRobot[];
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
    const { formData, updateField, activeType } = useRobotFormFields<BimanualSO101FormData>();
    const robotsQuery = $api.useSuspenseQuery('get', '/api/projects/{project_id}/robots', {
        params: { path: { project_id } },
    });
    const sourceType =
        activeType === ('BimanualSO101_Leader' as ConfigurableRobotType) ? 'SO101_Leader' : 'SO101_Follower';
    const eligibleRobots = robotsQuery.data.filter((robot) => isEligibleSource(robot, sourceType));
    const leftRobots = eligibleRobots.filter(
        (robot) => robot.payload.serial_number !== formData.payload.right_serial_number
    );
    const rightRobots = eligibleRobots.filter(
        (robot) => robot.payload.serial_number !== formData.payload.left_serial_number
    );

    const selectArm = (arm: 'left' | 'right', robotId: string | number | null) => {
        const robot = eligibleRobots.find(({ id }) => id === robotId);
        if (robot === undefined || robot.payload.calibration == null) {
            return;
        }

        if (arm === 'left') {
            updateField('payload', {
                ...formData.payload,
                left_serial_number: robot.payload.serial_number,
                left_calibration: robot.payload.calibration,
            });
            return;
        }

        updateField('payload', {
            ...formData.payload,
            right_serial_number: robot.payload.serial_number,
            right_calibration: robot.payload.calibration,
        });
    };

    const selectedRobotId = (arm: 'left' | 'right') => {
        const serial_number =
            arm === 'left' ? formData.payload.left_serial_number : formData.payload.right_serial_number;
        return eligibleRobots.find((robot) => robot.payload.serial_number === serial_number)?.id ?? null;
    };

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
