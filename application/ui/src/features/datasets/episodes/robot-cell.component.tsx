import { useEffect, useState } from 'react';

import { View } from '@geti-ui/ui';

import { RobotViewer } from '../../robots/controller/robot-viewer';
import { RobotModelsProvider } from '../../robots/robot-models-context';
import { useEpisodeViewer } from './episode-viewer-provider.component';

const InnerCell = ({ robotId }: { robotId: string }) => {
    const { environment, episode, player } = useEpisodeViewer();

    const [frameIndex, setFrameIndex] = useState<number>(0);

    useEffect(() => {
        if (player.isPlaying || player.isSeeking) {
            const interval = setInterval(() => {
                setFrameIndex(Math.floor(player.timeRef.current * episode.fps));
            }, 1000 / 60);
            return () => clearInterval(interval);
        }
    }, [player, episode]);

    const robot = environment.robots?.find((r) => r.robot.id === robotId)?.robot;
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
