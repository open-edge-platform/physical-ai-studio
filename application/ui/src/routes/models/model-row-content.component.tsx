import { Heading, Item, TabList, TabPanels, Tabs, View } from '@geti-ui/ui';

import { SchemaModel } from '../../api/openapi-spec';
import { MetricsContent } from './metrics';

import classes from './model-row-content.module.scss';

interface ModelRowContentProps {
    model: SchemaModel;
}

export const ModelRowContent = ({ model }: ModelRowContentProps) => {
    return (
        <View UNSAFE_className={classes.modelRowContent}>
            <Tabs>
                <TabList>
                    <Item key='metrics'>Model Metrics</Item>
                    <Item key='datasets'>Training Datasets</Item>
                    <Item key='export'>Export</Item>
                </TabList>
                <TabPanels>
                    <Item key='metrics'>
                        <MetricsContent modelId={model.id!} />
                    </Item>
                    <Item key='datasets'>
                        <Heading>Coming soon</Heading>
                    </Item>
                    <Item key='export'>
                        <Heading>Coming soon</Heading>
                    </Item>
                </TabPanels>
            </Tabs>
        </View>
    );
};
