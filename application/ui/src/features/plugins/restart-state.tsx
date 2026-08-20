import { createContext, ReactNode, useContext, useEffect, useMemo, useState } from 'react';

import {
    Button,
    ButtonGroup,
    Content,
    Dialog,
    DialogContainer,
    Divider,
    Flex,
    Heading,
    ProgressCircle,
    Text,
} from '@geti-ui/ui';
import { useQueryClient } from '@tanstack/react-query';

import { $api } from '../../api/client';
import { SchemaTrainJob } from '../../api/openapi-spec';

type HealthResponse = {
    status?: string;
    instance_id?: string;
    restart_required?: boolean;
};

type RestartStateValue = {
    restartRequired: boolean;
    restartStatus: 'idle' | 'restarting' | 'failed';
    restartPromptOpen: boolean;
    activeTrainingJobCount: number;
    hasActiveTrainingJobs: boolean;
    triggerRestartRequired: () => void;
    openRestartPrompt: () => void;
    closeRestartPrompt: () => void;
    restartServer: () => Promise<void>;
};

const HEALTH_POLL_INTERVAL_MS = 1500;
const MIN_RESTART_DIALOG_MS = 2000;

const RestartStateContext = createContext<RestartStateValue | null>(null);

export const RestartStateProvider = ({ children }: { children: ReactNode }) => {
    const queryClient = useQueryClient();
    const [restartRequested, setRestartRequested] = useState(false);
    const [restartPromptOpen, setRestartPromptOpen] = useState(false);
    const [previousInstanceId, setPreviousInstanceId] = useState<string>();

    const restartMutation = $api.useMutation('post', '/api/system/restart', {
        meta: { skipInvalidation: true },
    });
    const { data: jobs = [] } = $api.useQuery('get', '/api/jobs');
    const healthQuery = $api.useQuery('get', '/api/health', {
        retry: false,
        staleTime: 0,
        refetchOnWindowFocus: false,
        refetchOnReconnect: 'always',
    });
    const health = healthQuery.data as HealthResponse | undefined;
    const isRestarting = previousInstanceId !== undefined;
    const restartRequired = restartRequested || health?.restart_required === true;

    const activeTrainingJobCount = jobs.filter(
        (job): job is SchemaTrainJob =>
            job.type === 'training' && (job.status === 'running' || job.status === 'pending')
    ).length;

    const triggerRestartRequired = () => {
        setRestartRequested(true);
    };

    const openRestartPrompt = () => {
        setRestartPromptOpen(true);
    };

    const closeRestartPrompt = () => {
        if (!isRestarting) {
            setRestartPromptOpen(false);
        }
    };

    const restartServer = async () => {
        if (isRestarting) {
            return;
        }

        const { data } = await healthQuery.refetch();
        const instanceId = (data as HealthResponse | undefined)?.instance_id;
        if (instanceId === undefined) {
            return;
        }

        setRestartPromptOpen(true);
        setPreviousInstanceId(instanceId);

        try {
            await restartMutation.mutateAsync({});
        } catch {
            // The backend may replace itself before the response is returned.
        }
    };

    useEffect(() => {
        if (!isRestarting || health?.instance_id === previousInstanceId || health?.restart_required !== false) {
            return;
        }

        const remainingMs = Math.max(0, restartMutation.submittedAt + MIN_RESTART_DIALOG_MS - Date.now());
        const timeout = window.setTimeout(() => {
            queryClient.clear();
            setPreviousInstanceId(undefined);
            setRestartRequested(false);
            setRestartPromptOpen(false);
        }, remainingMs);

        return () => window.clearTimeout(timeout);
    }, [
        health?.instance_id,
        health?.restart_required,
        isRestarting,
        previousInstanceId,
        queryClient,
        restartMutation.submittedAt,
    ]);

    useEffect(() => {
        if (!isRestarting || healthQuery.isFetching) {
            return;
        }

        const timeout = window.setTimeout(() => {
            void healthQuery.refetch();
        }, HEALTH_POLL_INTERVAL_MS);

        return () => window.clearTimeout(timeout);
    }, [
        healthQuery.dataUpdatedAt,
        healthQuery.errorUpdatedAt,
        healthQuery.isFetching,
        healthQuery.refetch,
        isRestarting,
    ]);

    const value = useMemo(
        (): RestartStateValue => ({
            restartRequired,
            restartStatus: isRestarting ? 'restarting' : 'idle',
            restartPromptOpen,
            activeTrainingJobCount,
            hasActiveTrainingJobs: activeTrainingJobCount > 0,
            triggerRestartRequired,
            openRestartPrompt,
            closeRestartPrompt,
            restartServer,
        }),
        [activeTrainingJobCount, isRestarting, restartPromptOpen, restartRequired]
    );

    return (
        <RestartStateContext.Provider value={value}>
            {children}
            {restartRequired && restartPromptOpen ? (
                <DialogContainer onDismiss={closeRestartPrompt}>
                    <Dialog>
                        <Heading>Restart server now?</Heading>
                        <Divider />
                        <Content>
                            <Flex direction='column' gap='size-150'>
                                <Text>Plugin changes require a server restart to become active.</Text>
                                {activeTrainingJobCount > 0 ? (
                                    <Text>
                                        Restarting now will interrupt {activeTrainingJobCount} active training job
                                        {activeTrainingJobCount === 1 ? '' : 's'}.
                                    </Text>
                                ) : null}
                                {isRestarting ? (
                                    <Flex alignItems='center' gap='size-100'>
                                        <ProgressCircle aria-label='Restarting server' isIndeterminate size='S' />
                                        <Text>Waiting for server restart…</Text>
                                    </Flex>
                                ) : null}
                            </Flex>
                        </Content>
                        <ButtonGroup>
                            <Button variant='accent' isDisabled={isRestarting} onPress={() => void restartServer()}>
                                {isRestarting ? 'Restarting…' : 'Restart now'}
                            </Button>
                            {!isRestarting ? (
                                <Button variant='secondary' onPress={closeRestartPrompt}>
                                    Later
                                </Button>
                            ) : null}
                        </ButtonGroup>
                    </Dialog>
                </DialogContainer>
            ) : null}
        </RestartStateContext.Provider>
    );
};

export const useRestartState = () => {
    const context = useContext(RestartStateContext);
    if (context === null) {
        throw new Error('useRestartState must be used within RestartStateProvider');
    }
    return context;
};
