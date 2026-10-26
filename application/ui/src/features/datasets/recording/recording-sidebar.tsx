import { Flex } from '@geti-ui/ui';

import { EpisodeControls } from './episode-controls';
import { EpisodeSection } from './episode-section';
import { RobotsSection } from './robots-section';
import { TaskPicker } from './task-picker';

export const RecordingSidebar = () => (
    <Flex direction='column' gap='size-300'>
        <EpisodeSection />
        <TaskPicker />
        <EpisodeControls />
        <RobotsSection />
    </Flex>
);
