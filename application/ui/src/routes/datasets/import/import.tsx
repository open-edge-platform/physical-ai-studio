import { Button, Flex, Heading, Text, View } from '@geti/ui';
import { clsx } from 'clsx';

import { $api } from '../../../api/client';
import { SchemaLeRobotDatasetInfo } from '../../../api/openapi-spec';
import { useProjectId } from '../../../features/projects/use-project';

import classes from './import.module.scss';
import { paths } from '../../../router';

interface ImportableDatasetProps {
    dataset: SchemaLeRobotDatasetInfo;
}
const ImportableDataset = ({ dataset }: ImportableDatasetProps) => {
    const { project_id } = useProjectId();
    const importDatasetMutation = $api.useMutation('post', '/api/projects/{project_id}/import_dataset');

    const importDataset = () => {
        importDatasetMutation.mutateAsync({
            params: {
                path: { project_id },
            },
            body: dataset,
        });
    };

    const cameras = dataset.features
        .map((name) => {
            return [...name.matchAll(/\.images\.(\w*)/g)].at(0)?.at(1);
        })
        .filter((m) => m);
    return (
        <Flex UNSAFE_className={clsx({ [classes.card]: true })} alignItems='center' maxWidth={'size-5000'}>
            <View width={'100%'} padding={'size-200'}>
                <Flex alignItems={'center'} justifyContent={'space-between'}>
                    <Heading level={3}>{dataset.repo_id}</Heading>
                </Flex>

                <Flex gap={'size-100'} direction={'column'}>
                    <Text>• Cameras: {cameras.join(', ')}</Text>
                </Flex>
            </View>
            <View padding={'size-200'}>
                <Button onPress={importDataset} isPending={importDatasetMutation.isPending}>
                    Import
                </Button>
            </View>
        </Flex>
    );
};

export const ImportDataset = () => {
    const { project_id } = useProjectId();
    const { data: lerobotDatasets } = $api.useSuspenseQuery('get', '/api/dataset/lerobot');
    const { data: orphanDatasets } = $api.useSuspenseQuery('get', '/api/dataset/orphans');

    return (
        <View>
            <Heading>Import dataset</Heading>
            <Button href={paths.project.datasets.record_new({project_id})} >Setup</Button>
            {[...lerobotDatasets, ...orphanDatasets].map((dataset) => (
                <ImportableDataset dataset={dataset} key={dataset.repo_id} />
            ))}
        </View>
    );
};
