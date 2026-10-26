import { Badge, Flex, Image, Text, View } from '@geti-ui/ui';

import RobotArm from '../../../assets/robot-arm.webp';
import { isUnavailableRobot } from '../../robots/robot-types';
import { useRuntimeSession } from '../../robots/runtime-session-provider';
import { sessionForRobot, useRuntimeSessions } from '../../runtime-sessions/use-runtime-sessions';
import {
    getFollowerStatus,
    getLeaderStatus,
    getRobotSubtitle,
    getSessionRobots,
    RobotStatus,
    SessionRobot,
    UNAVAILABLE_ROBOT_STATUS,
} from './robot-status';
import { SidebarSection } from './sidebar-section';

import classes from './recording-sidebar.module.css';

interface RobotCardProps {
    sessionRobot: SessionRobot;
    status: RobotStatus;
}

const RobotCard = ({ sessionRobot, status }: RobotCardProps) => (
    <View
        backgroundColor='gray-75'
        borderWidth='thin'
        borderColor='gray-300'
        borderRadius='small'
        paddingY='size-100'
        paddingX='size-150'
        minWidth={0}
    >
        <Flex alignItems='center' gap='size-150'>
            <Image src={RobotArm} alt='' width='size-400' flexShrink={0} />
            <Flex direction='column' flexGrow={1} minWidth={0}>
                <Text UNSAFE_className={classes.robotName}>{sessionRobot.robot.name}</Text>
                <Text UNSAFE_className={classes.robotSubtitle}>{getRobotSubtitle(sessionRobot)}</Text>
            </Flex>
            <Badge variant={status.variant} flexShrink={0}>
                {status.label}
            </Badge>
        </Flex>
    </View>
);

export const RobotsSection = () => {
    const { environment, state } = useRuntimeSession();
    const { data: sessions } = useRuntimeSessions();

    const robots = getSessionRobots(environment);

    if (robots.length === 0) {
        return null;
    }

    const getStatus = (sessionRobot: SessionRobot): RobotStatus => {
        if (isUnavailableRobot(sessionRobot.robot)) {
            return UNAVAILABLE_ROBOT_STATUS;
        }

        return sessionRobot.role === 'follower'
            ? getFollowerStatus(state, sessionForRobot(sessions, sessionRobot.robot.id))
            : getLeaderStatus(state);
    };

    return (
        <SidebarSection label='Robots'>
            {robots.map((sessionRobot) => (
                <RobotCard
                    key={`${sessionRobot.role}-${sessionRobot.robot.id}`}
                    sessionRobot={sessionRobot}
                    status={getStatus(sessionRobot)}
                />
            ))}
        </SidebarSection>
    );
};
