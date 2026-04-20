import { ActionButton, Button, DialogTrigger, Flex, Grid, Item, Key, Menu, MenuTrigger, TabList, Tabs, Text, View, TabPanels } from '@geti-ui/ui';
import { SchemaModel } from "../../api/openapi-spec"
import { MetricsContent } from './metrics';

import classes from './model-row-content.module.scss'

interface ModelRowContentProps {
    model: SchemaModel;
}

export const ModelRowContent = ({ model }: ModelRowContentProps) => {
    return (
        <View UNSAFE_className={classes.modelRowContent}>
            <Tabs>
                <TabList>
                    <Item key="metrics">Model Metrics</Item>
                </TabList>
                <TabPanels>
                    <Item key="metrics">
                        <MetricsContent modelId={model.id!} />
                    </Item>
                </TabPanels>
            </Tabs>
        </View>
    );
}
