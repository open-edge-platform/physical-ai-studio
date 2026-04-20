import { ActionButton, Button, DialogTrigger, Flex, Grid, Item, Key, Menu, MenuTrigger, TabList, Tabs, Text, View, TabPanels, Heading } from '@geti-ui/ui';
import { SchemaTrainJob } from "../../api/openapi-spec"
import { JobMetricsContent, MetricsContent } from './metrics';

import classes from './model-row-content.module.scss'

interface JobRowContentProps {
    job: SchemaTrainJob;
}

export const JobRowContent = ({ job }: JobRowContentProps) => {
    return (
        <View UNSAFE_className={classes.modelRowContent}>
            <Tabs>
                <TabList>
                    <Item key="metrics">Model Metrics</Item>
                    <Item key="datasets">Training Datasets</Item>
                </TabList>
                <TabPanels>
                    <Item key="metrics">
                        <JobMetricsContent jobId={job.id!} />
                    </Item>
                    <Item key="datasets">
                      <Heading>Coming soon</Heading>
                    </Item>
                </TabPanels>
            </Tabs>
        </View>
    );
}
