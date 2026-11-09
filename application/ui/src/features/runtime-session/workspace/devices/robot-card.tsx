import { Image } from '@geti-ui/ui';

import { SchemaStatusMessage } from '../../../../api/openapi-spec';
import RobotArm from '../../../../assets/robot-arm.webp';
import { isUnavailableRobot, SchemaRobot } from '../../../robots/robot-types';
import { useRuntimeSession } from '../../runtime-session-provider';
import { DeviceCard } from './device-card';
import { getRobotStatus, getRobotSubtitle, RobotRole, UNAVAILABLE_ROBOT_STATUS } from './session-devices';

interface RobotCardProps {
    role: RobotRole;
    robot: SchemaRobot;
    status: SchemaStatusMessage;
}

export const RobotCard = ({ role, robot, status }: RobotCardProps) => {
    const { state } = useRuntimeSession();

    return (
        <DeviceCard
            thumb={<Image src={RobotArm} alt='' width='size-400' flexShrink={0} />}
            title={robot.name}
            subtitle={getRobotSubtitle(robot, role)}
            status={isUnavailableRobot(robot) ? UNAVAILABLE_ROBOT_STATUS : getRobotStatus(role, status, state)}
            error={status.error}
        />
    );
};
