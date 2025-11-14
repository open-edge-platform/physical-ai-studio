import { Suspense, useState } from "react";
import { InferenceViewer } from "./inference-viewer"
import { SchemaInferenceConfig } from "../../../api/openapi-spec";
import { InferenceSetup } from "../../../features/configuration/inference/inference-setup";
import { useInferenceParams } from "./use-inference-params";
import { View, Well, Flex, Heading, Divider, Content, Loading } from "@geti/ui";

export const Index = () => {
    const { model_id } = useInferenceParams();
    const [config, setConfig] = useState<SchemaInferenceConfig>()


    if (!config) {
        return (
            <Flex flex width="100%" justifyContent={'center'}>
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
        )
    } else {
        return (
            <InferenceViewer config={config} />
        );
    }
}
