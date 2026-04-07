import { View } from '@geti/ui';

import { useEpisodeViewer } from './episode-viewer-provider.component';
import { RobotViewer } from '../../robots/controller/robot-viewer';
import { RobotModelsProvider } from '../../robots/robot-models-context';

const InnerCell = ({ robotId }: { robotId: string }) => {
    const { environment, episode, player } = useEpisodeViewer();

    const frameIndex = Math.floor(player.time * episode.fps);

    const robot = environment.robots?.find((r) => r.robot.id === robotId)?.robot
    if (robot === undefined) {
        return <></>;
    }

    return (
        <View minWidth='size-4000' minHeight='size-4000' width='100%' height='100%' backgroundColor={'gray-600'}>
            <RobotViewer
                key={robotId}
                featureValues={episode.actions[frameIndex]}
                featureNames={episode.action_keys}
                robot={robot}
            />
        </View>
    );
};

export const RobotCell = ({ robotId }: { robotId: string }) => {
    return (
        <RobotModelsProvider>
            <InnerCell robotId={robotId} />
        </RobotModelsProvider>
    );
};
