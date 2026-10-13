import { Suspense } from 'react';

import { Flex, Grid, Icon, Link, Loading, minmax, Text, View } from '@geti-ui/ui';
import { ChevronLeft } from '@geti-ui/ui/icons';

import { $api } from '../../../api/client';
import { SchemaEnvironmentDraft, SchemaModel } from '../../../api/openapi-spec';
import { AppFooter } from '../../../components/app-footer/app-footer';
import { paths } from '../../../paths';
import { RuntimeSessionProvider } from '../../runtime-session/runtime-session-provider';
import { SessionWorkspace } from '../../runtime-session/workspace/session-workspace';
import { useInferenceParams } from './use-inference-params';

import classes from '../../../routes/datasets/record/index.module.css';

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
    const { model_id } = useInferenceParams();
    const {
        data: { model },
    } = $api.useSuspenseQuery('get', '/api/models/{model_id}', {
        params: { path: { model_id } },
    });

    // Datasets do not select an environment; the side bar loads one for this session.
    return <InferenceWorkspace model={model} environment={null} />;
};
