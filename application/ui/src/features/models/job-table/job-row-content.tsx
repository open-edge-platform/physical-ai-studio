import { Item, TabList, TabPanels, Tabs, View } from '@geti-ui/ui';

import { SchemaTrainJob } from '../../../api/openapi-spec';
import { JobMetricsContent } from '../metrics/metrics';
import { ComingSoon } from '../shared/coming-soon';

import classes from '../models-table/model-row-content.module.css';

interface JobRowContentProps {
    job: SchemaTrainJob;
}

export const JobRowContent = ({ job }: JobRowContentProps) => {
    return (
        <View UNSAFE_className={classes.modelRowContent}>
            <Tabs>
                <TabList marginBottom={'size-200'}>
                    <Item key='metrics'>Model Metrics</Item>
                    <Item key='datasets'>Training Datasets</Item>
                </TabList>
                <TabPanels>
                    <Item key='metrics'>
                        <JobMetricsContent jobId={job.id!} />
                    </Item>
                    <Item key='datasets'>
                        <ComingSoon />
                    </Item>
                </TabPanels>
            </Tabs>
        </View>
    );
};
