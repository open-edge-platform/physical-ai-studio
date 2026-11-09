import { $api } from '../../../../api/client';
import { SchemaStatusMessage } from '../../../../api/openapi-spec';
import { useProjectId } from '../../../projects/use-project';
import { useRuntimeSession } from '../../runtime-session-provider';
import { DeviceCard } from './device-card';
import { RobotCard } from './robot-card';
import { getRobotStatus, RobotRole } from './session-devices';

interface RobotsListProps {
    robots: { role: RobotRole; id: string; status: SchemaStatusMessage }[];
}

export const RobotsList = ({ robots }: RobotsListProps) => {
    const { project_id } = useProjectId();
    const { state } = useRuntimeSession();
    const { data: projectRobots } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/robots', {
        params: { path: { project_id } },
    });

    return (
        <>
            {robots.map(({ role, id, status }) => {
                const robot = projectRobots.find((candidate) => candidate.id === id);

                return robot === undefined ? (
                    <DeviceCard
                        key={role}
                        title={id}
                        subtitle='Unknown robot'
                        status={getRobotStatus(role, status, state)}
                        error={status.error}
                    />
                ) : (
                    <RobotCard key={role} role={role} robot={robot} status={status} />
                );
            })}
        </>
    );
};
