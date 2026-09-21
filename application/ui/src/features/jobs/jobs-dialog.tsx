import { useState } from 'react';

import {
    Badge,
    Button,
    Content,
    Dialog,
    Divider,
    Flex,
    Heading,
    IllustratedMessage,
    Item,
    Loading,
    TabList,
    TabPanels,
    Tabs,
    Text,
} from '@geti-ui/ui';

import { $api } from '../../api/client';
import { SchemaTrainJob } from '../../api/openapi-spec';
import { ReactComponent as EmptyIllustration } from '../../assets/illustration.svg';
import { TrainingJobsTable } from '../models/job-table/training-jobs-table';

import classes from './jobs-dialog.module.css';

export type JobStatusTabKey = SchemaTrainJob['status'] | 'all';

interface StatusTab {
    key: JobStatusTabKey;
    label: string;
    emptyMessage: string;
}

const STATUS_TABS: StatusTab[] = [
    { key: 'all', label: 'All jobs', emptyMessage: 'No jobs yet.' },
    { key: 'running', label: 'Running jobs', emptyMessage: 'No running jobs.' },
    { key: 'completed', label: 'Finished jobs', emptyMessage: 'No finished jobs.' },
    { key: 'pending', label: 'Scheduled jobs', emptyMessage: 'No scheduled jobs.' },
    { key: 'canceled', label: 'Cancelled jobs', emptyMessage: 'No cancelled jobs.' },
    { key: 'failed', label: 'Failed jobs', emptyMessage: 'No failed jobs.' },
];

const tabStatus = (key: JobStatusTabKey): SchemaTrainJob['status'] | undefined => (key === 'all' ? undefined : key);

const JobsEmptyState = ({ message }: { message: string }) => (
    <IllustratedMessage>
        <EmptyIllustration />
        <Heading>{message}</Heading>
    </IllustratedMessage>
);

interface JobsDialogProps {
    projectId?: string;
    onViewLogs: (job: SchemaTrainJob) => void;
    close: () => void;
    selectedTab?: JobStatusTabKey;
    onSelectedTabChange?: (tab: JobStatusTabKey) => void;
}

export const JobsDialog = ({ projectId, onViewLogs, close, selectedTab, onSelectedTabChange }: JobsDialogProps) => {
    const [internalTab, setInternalTab] = useState<JobStatusTabKey>(selectedTab ?? 'all');
    const effectiveTab = selectedTab ?? internalTab;

    const setSelectedTab = (tab: JobStatusTabKey) => {
        onSelectedTabChange?.(tab);
        if (selectedTab === undefined) {
            setInternalTab(tab);
        }
    };

    const { data, isPending, isError, refetch } = $api.useQuery('get', '/api/jobs');

    const trainingJobs = (data ?? [])
        .filter((job): job is SchemaTrainJob => job.type === 'training')
        .filter((job) => projectId === undefined || job.project_id === projectId);

    const heading = projectId === undefined ? 'All projects jobs' : 'Current project jobs';

    const counts = STATUS_TABS.reduce<Record<JobStatusTabKey, number>>(
        (acc, tab) => {
            const status = tabStatus(tab.key);
            acc[tab.key] =
                status === undefined ? trainingJobs.length : trainingJobs.filter((job) => job.status === status).length;
            return acc;
        },
        {} as Record<JobStatusTabKey, number>
    );

    return (
        <Dialog width={'90vw'} height={'70vh'} onDismiss={close} isDismissable>
            <Heading>{heading}</Heading>
            <Divider />
            <Content UNSAFE_className={classes.dialogContent}>
                {isPending ? (
                    <Flex alignItems='center' justifyContent='center' height='100%'>
                        <Loading mode='inline' />
                    </Flex>
                ) : isError && data === undefined ? (
                    <Flex direction='column' alignItems='center' justifyContent='center' height='100%' gap='size-100'>
                        <Text>Failed to load jobs.</Text>
                        <Button variant='secondary' onPress={() => refetch()}>
                            Retry
                        </Button>
                    </Flex>
                ) : (
                    <Flex direction='column' height='100%' minHeight={0}>
                        {isError && (
                            <Flex alignItems='center' gap='size-100' marginBottom='size-100'>
                                <Text>Failed to refresh jobs.</Text>
                                <Button variant='secondary' onPress={() => refetch()}>
                                    Retry
                                </Button>
                            </Flex>
                        )}
                        <Tabs
                            selectedKey={effectiveTab}
                            onSelectionChange={(key) => setSelectedTab(key as JobStatusTabKey)}
                            height='100%'
                            minHeight={0}
                        >
                            <TabList aria-label='Job status'>
                                {STATUS_TABS.map((tab) => {
                                    const count = counts[tab.key];
                                    return (
                                        <Item
                                            key={tab.key}
                                            textValue={count > 0 ? `${tab.label} (${count})` : tab.label}
                                        >
                                            <Flex alignItems='center' gap='size-100'>
                                                <Text>{tab.label}</Text>
                                                {count > 0 && (
                                                    <Badge
                                                        variant='neutral'
                                                        UNSAFE_style={
                                                            tab.key === effectiveTab
                                                                ? { backgroundColor: 'var(--energy-blue)' }
                                                                : undefined
                                                        }
                                                    >
                                                        {count}
                                                    </Badge>
                                                )}
                                            </Flex>
                                        </Item>
                                    );
                                })}
                            </TabList>
                            <TabPanels minHeight={0} UNSAFE_className={classes.tabPanels}>
                                {STATUS_TABS.map((tab) => (
                                    <Item key={tab.key}>
                                        <TrainingJobsTable
                                            jobs={trainingJobs.filter((job) => {
                                                const status = tabStatus(tab.key);
                                                return status === undefined || job.status === status;
                                            })}
                                            onViewLogs={onViewLogs}
                                            emptyMessage={<JobsEmptyState message={tab.emptyMessage} />}
                                        />
                                    </Item>
                                ))}
                            </TabPanels>
                        </Tabs>
                    </Flex>
                )}
            </Content>
        </Dialog>
    );
};
