import { $api } from '../../api/client';
import { ErrorMessage } from '../../components/error-page/error-page';
import { useProject } from '../projects/project.provider';
import { View, Text, Flex, Divider, Well } from '@geti/ui'
import { DatasetViewer } from './dataset-viewer';
import { DatasetProvider } from './dataset.provider';

export const Index = () => {
    const { project } = useProject();
    const dataset = project.datasets[0];
    return (
        <Flex direction={'column'} height={'100%'}>
            <View>
                {project.datasets.map((dataset) => (
                    <Text key={dataset}>{dataset}</Text>
                ))}
            </View>
            <Flex flex={1}>
                <Well flex={1}>
                    {dataset === undefined
                        ? <Text>No datasets yet...</Text>
                        : (
                            <DatasetProvider project_id={project.id} repo_id={dataset}>
                                <DatasetViewer />
                            </DatasetProvider>
                            )
                    }
                </Well>
            </Flex>

        </Flex>
    )
};
