import { Button, Flex, Heading, ProgressCircle, StatusLight, Text } from '@geti-ui/ui';

import { paths } from '../../../paths';
import { useRuntimeSession } from '../../robots/runtime-session-provider';

export const Initializing = () => {
    const { dataset, state } = useRuntimeSession();

    return (
        <Flex width='100%' height='100%' alignItems='center' justifyContent='center' direction='column'>
            <Heading level={2}>
                <Text>Initializing</Text>
                <ProgressCircle marginStart='size-200' size='S' isIndeterminate alignSelf='center' />
            </Heading>
            <Flex direction='column' margin='size-200'>
                <StatusLight variant={state.dataset_loaded ? 'positive' : 'yellow'}>Dataset</StatusLight>
                <StatusLight variant={state.connected ? 'positive' : 'yellow'}>Environment</StatusLight>
            </Flex>
            {dataset?.id !== undefined && (
                <Button
                    variant='secondary'
                    href={paths.project.datasets.show({ dataset_id: dataset.id, project_id: dataset.project_id })}
                >
                    Cancel
                </Button>
            )}
        </Flex>
    );
};
