import { Button, Flex, Icon, Text, View } from '@geti-ui/ui';
import { ChevronLeft } from '@geti-ui/ui/icons';
import { Link } from 'react-router';

import { paths } from '../../../paths';
import { useRuntimeSession } from '../../robots/runtime-session-provider';
import { formatEpisodeCounts } from './format';
import { useRecording } from './recording-provider';
import { useRecordingEpisodes } from './use-recording-episodes';

import classes from './recording-page.module.css';

const BackContent = ({ subtitle }: { subtitle: string }) => (
    <Flex alignItems='center' gap='size-200'>
        <Icon>
            <ChevronLeft />
        </Icon>
        <Flex direction='column'>
            <Text UNSAFE_className={classes.headerText}>Recording episode</Text>
            <Text UNSAFE_className={classes.subHeaderText}>{subtitle}</Text>
        </Flex>
    </Flex>
);

export const RecordingTopBar = () => {
    const { dataset, environment, state } = useRuntimeSession();
    const { task } = useRecording();
    const { episodesInDataset, episodesThisSession } = useRecordingEpisodes();

    if (dataset?.id === undefined) {
        throw new Error('Cannot render the recording top bar without a dataset.');
    }

    const datasetUrl = paths.project.datasets.show({ project_id: dataset.project_id, dataset_id: dataset.id });
    const subtitle = [dataset.name, task, environment.name].filter((part) => part !== '').join(' · ');
    const counts = formatEpisodeCounts(episodesInDataset, episodesThisSession);

    return (
        <View backgroundColor='gray-300' gridArea='header'>
            <Flex height='100%' alignItems='center' marginX='1rem' gap='size-200' justifyContent='space-between'>
                {state.is_recording ? (
                    <span className={classes.headerLink} aria-disabled='true'>
                        <BackContent subtitle={subtitle} />
                    </span>
                ) : (
                    <Link to={datasetUrl} className={classes.headerLink} aria-label='Back to dataset'>
                        <BackContent subtitle={subtitle} />
                    </Link>
                )}
                <Flex alignItems='center' gap='size-200'>
                    {counts !== null && <Text UNSAFE_className={classes.episodesText}>{counts}</Text>}
                    <Button variant='secondary' href={datasetUrl} isDisabled={state.is_recording}>
                        Done
                    </Button>
                </Flex>
            </Flex>
        </View>
    );
};
