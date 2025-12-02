import { Suspense, useState } from 'react';

import { Content, Divider, Flex, Heading, Loading, Well } from '@geti/ui';

import { SchemaInferenceConfig } from '../../../api/openapi-spec';
import { InferenceSetup } from '../../../features/configuration/inference/inference-setup';
import { InferenceViewer } from './inference-viewer';
import { useInferenceParams } from './use-inference-params';

export const Index = () => {
    const { model_id } = useInferenceParams();
    const [config, setConfig] = useState<SchemaInferenceConfig>();

    if (!config) {
        return (
            <Flex flex width='100%' justifyContent={'center'}>
                <Well margin={'size-200'} width={'600px'}>
                    <Heading>Inference Setup</Heading>
                    <Divider />
                    <Content>
                        <Suspense fallback={<Loading mode='overlay' />}>
                            <InferenceSetup model_id={model_id} onDone={setConfig} />
                        </Suspense>
                    </Content>
                </Well>
            </Flex>
        );
    } else {
        return <InferenceViewer config={config} />;
    }
};
