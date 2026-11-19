import { ReactNode, Suspense, useState } from 'react';

import {
    Content,
    Flex,
    Grid,
    Heading,
    Icon,
    IllustratedMessage,
    Item,
    Link,
    Loading,
    minmax,
    Picker,
    Text,
    View,
} from '@geti-ui/ui';
import { ChevronLeft } from '@geti-ui/ui/icons';

import { $api } from '../../../api/client';
import { SchemaEnvironmentDraft, SchemaEnvironmentWithRelations, SchemaModel } from '../../../api/openapi-spec';
import { AppFooter } from '../../../components/app-footer/app-footer';
import { paths } from '../../../paths';
import { useRecordingEnvironmentsQuery } from '../../datasets/recording-environments/use-dataset-environment';
import { environmentDraft, RuntimeSessionProvider } from '../../runtime-session/runtime-session-provider';
import { SessionWorkspace } from '../../runtime-session/workspace/session-workspace';
import { useInferenceParams } from './use-inference-params';

import classes from '../../../routes/datasets/record/index.module.css';

interface InferenceEnvironmentProps {
    projectId: string;
    environmentId: string;
    children: (environment: SchemaEnvironmentWithRelations) => ReactNode;
}

const InferenceEnvironment = ({ projectId, environmentId, children }: InferenceEnvironmentProps) => {
    const { data: environment } = $api.useSuspenseQuery(
        'get',
        '/api/projects/{project_id}/environments/{environment_id}',
        { params: { path: { project_id: projectId, environment_id: environmentId } } }
    );

    return children(environment);
};

const InferenceSession = ({ model }: { model: SchemaModel }) => (
    <SessionWorkspace
        initialMode='inference'
        cancelHref={paths.project.models.index({ project_id: model.project_id })}
    />
);

/** The model running in a session with a user-selected environment. */
const InferenceWorkspace = ({
    model,
    environment,
}: {
    model: SchemaModel;
    environment: SchemaEnvironmentDraft | null;
}) => {
    const { project_id, backend, device } = useInferenceParams();

    return (
        <RuntimeSessionProvider
            project_id={project_id}
            environment={environment}
            model={{ model_id: model.id!, backend, device: device ?? 'auto' }}
        >
            <Grid
                areas={['header', 'content', 'footer']}
                UNSAFE_style={{
                    // eslint-disable-next-line max-len
                    gridTemplateRows: `var(--spectrum-global-dimension-size-800, 4rem) ${minmax(0, '1fr')} var(--spectrum-global-dimension-size-400)`,
                }}
                minHeight={0}
                height={'100%'}
            >
                <View backgroundColor={'gray-300'} gridArea={'header'}>
                    <Flex height='100%' alignItems={'center'} marginX='1rem'>
                        <Link href={paths.project.models.index({ project_id })} isQuiet variant='overBackground'>
                            <div>
                                <Flex direction='row' gap='size-200' alignItems={'center'}>
                                    <Icon>
                                        <ChevronLeft />
                                    </Icon>
                                    <Flex direction={'column'}>
                                        <Text UNSAFE_className={classes.headerText}>Running model</Text>
                                        <Text UNSAFE_className={classes.subHeaderText}>{model.name}</Text>
                                    </Flex>
                                </Flex>
                            </div>
                        </Link>
                    </Flex>
                </View>
                <View gridArea={'content'} maxHeight={'100vh'} minHeight={0} height='100%'>
                    <View padding='size-200' height='100%'>
                        {/* Inside the provider: anything that suspends while loading must not take the
                            page-level boundary, which would unmount the provider and end the session. */}
                        <Suspense fallback={<Loading mode='inline' />}>
                            <InferenceSession model={model} />
                        </Suspense>
                    </View>
                </View>
                <AppFooter />
            </Grid>
        </RuntimeSessionProvider>
    );
};

export const InferencePage = () => {
    const { project_id, model_id } = useInferenceParams();
    const [selectedEnvironmentId, setSelectedEnvironmentId] = useState<string>();
    const {
        data: { model },
    } = $api.useSuspenseQuery('get', '/api/models/{model_id}', {
        params: { path: { model_id } },
    });

    const { data: environments } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/environments', {
        params: { path: { project_id } },
    });
    const associations = useRecordingEnvironmentsQuery(model.dataset_id);
    const associatedEnvironments = associations.data?.filter(({ environment_id }) =>
        environments.some(({ id }) => id === environment_id)
    );
    // Only an unambiguous recording association can select hardware automatically.
    const associatedEnvironmentId =
        associatedEnvironments?.length === 1 ? associatedEnvironments[0].environment_id : undefined;
    const environmentId = selectedEnvironmentId ?? associatedEnvironmentId;

    if (associations.isLoading) return <Loading />;
    if (associations.isError) throw associations.error;

    if (environmentId === undefined) {
        return (
            <IllustratedMessage>
                <Heading>Select an inference environment</Heading>
                <Content>
                    {environments.length === 0 ? (
                        'Create an environment before starting inference.'
                    ) : (
                        <Picker
                            label='Environment'
                            items={environments}
                            selectedKey={null}
                            onSelectionChange={(id) => setSelectedEnvironmentId(id?.toString())}
                        >
                            {(environment) => <Item key={environment.id}>{environment.name}</Item>}
                        </Picker>
                    )}
                </Content>
            </IllustratedMessage>
        );
    }

    return (
        <InferenceEnvironment projectId={project_id} environmentId={environmentId}>
            {(environment) => <InferenceWorkspace model={model} environment={environmentDraft(environment)} />}
        </InferenceEnvironment>
    );
};
