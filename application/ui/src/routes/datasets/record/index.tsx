import { Suspense, useState } from 'react';

import { Flex, Grid, Icon, Link, Loading, Text, View } from '@geti/ui';
import { ChevronLeft } from '@geti/ui/icons';
import { useNavigate } from 'react-router';

import { $api } from '../../../api/client';
import { SchemaTeleoperationConfig } from '../../../api/openapi-spec';
import { TeleoperationSetup } from '../../../features/configuration/teleoperation/teleoperation';
import { useDatasetId } from '../../../features/datasets/use-dataset';
import { paths } from '../../../router';
import { RecordingViewer } from './recording-viewer';

import classes from './index.module.scss';

const RecordingPage = () => {
    const { project_id, dataset_id } = useDatasetId();
    const [recordingConfig, setRecordingConfig] = useState<SchemaTeleoperationConfig>();

    const navigate = useNavigate();

    const { data: dataset } = $api.useSuspenseQuery('get', '/api/dataset/{dataset_id}', {
        params: {
            path: {
                dataset_id,
            },
        },
    });
    const backPath = paths.project.datasets.show({ project_id, dataset_id });

    const onSetupDone = (config: SchemaTeleoperationConfig | undefined) => {
        if (config) {
            setRecordingConfig(config);
        } else {
            navigate(backPath);
        }
    };

    const subHeader = recordingConfig ? `${recordingConfig.environment.name} | ${recordingConfig.task}` : 'Setup';

    return (
        <Grid
            areas={['header', 'content']}
            UNSAFE_style={{
                gridTemplateRows: 'var(--spectrum-global-dimension-size-800, 4rem) auto',
            }}
            minHeight={0}
            height={'100%'}
        >
            <View backgroundColor={'gray-300'} gridArea={'header'}>
                <Flex height='100%' alignItems={'center'} marginX='1rem' gap='size-200'>
                    <Link href={backPath} isQuiet variant='overBackground'>
                        <Flex marginEnd='size-200' direction='row' gap='size-200' alignItems={'center'}>
                            <Icon>
                                <ChevronLeft />
                            </Icon>
                            <Flex direction={'column'}>
                                <Text UNSAFE_className={classes.headerText}>Adding Episode</Text>
                                <Text UNSAFE_className={classes.subHeaderText}>{subHeader}</Text>
                            </Flex>
                        </Flex>
                    </Link>
                </Flex>
            </View>

            <View gridArea={'content'} maxHeight={'100vh'} minHeight={0} height='100%'>
                {recordingConfig ? (
                    <View padding='size-200' height='100%'>
                        <RecordingViewer recordingConfig={recordingConfig} />
                    </View>
                ) : (
                    <Flex margin={'size-200'} justifySelf='center' flex maxWidth={'size-6000'}>
                        <TeleoperationSetup dataset={dataset} onDone={onSetupDone} />
                    </Flex>
                )}
            </View>
        </Grid>
    );
};

export const Index = () => {
    return (
        <Suspense fallback={<Loading mode='overlay' />}>
            <RecordingPage />
        </Suspense>
    );
};
