import { createContext, ReactNode, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';

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
import { pluralize } from '../../utils';

type RestartStateValue = {
    restartRequired: boolean;
    isRestarting: boolean;
    restartPromptOpen: boolean;
    activeTrainingJobCount: number;
    hasActiveTrainingJobs: boolean;
    triggerRestartRequired: () => void;
    openRestartPrompt: () => void;
    closeRestartPrompt: () => void;
    restartServer: () => Promise<void>;
};

const HEALTH_POLL_INTERVAL_MS = 1500;

const RestartStateContext = createContext<RestartStateValue | null>(null);

type RestartPromptDialogProps = {
    isRestarting: boolean;
    activeTrainingJobCount: number;
    onDismiss: () => void;
    onRestart: () => Promise<void>;
};

const RestartPromptDialog = ({
    isRestarting,
    activeTrainingJobCount,
    onDismiss,
    onRestart,
}: RestartPromptDialogProps) => (
    <DialogContainer onDismiss={onDismiss}>
        <Dialog>
            <Heading>Restart server now?</Heading>
            <Divider />
            <Content>
                <Flex direction='column' gap='size-150'>
                    <Text>Plugin changes require a server restart to become active.</Text>
                    {activeTrainingJobCount > 0 ? (
                        <Text>
                            Restarting now will interrupt {activeTrainingJobCount} active{' '}
                            {pluralize(activeTrainingJobCount, 'training job', 'training jobs')}.
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
                <Button variant='accent' isDisabled={isRestarting} onPress={() => void onRestart()}>
                    {isRestarting ? 'Restarting…' : 'Restart now'}
                </Button>
                {!isRestarting ? (
                    <Button variant='secondary' onPress={onDismiss}>
                        Later
                    </Button>
                ) : null}
            </ButtonGroup>
        </Dialog>
    </DialogContainer>
);

export const RestartStateProvider = ({ children }: { children: ReactNode }) => {
    const queryClient = useQueryClient();
    const [restartRequested, setRestartRequested] = useState(false);
    const [restartPromptOpen, setRestartPromptOpen] = useState(false);
    const [isRestarting, setIsRestarting] = useState(false);
    const lastInstanceId = useRef<string | undefined>(undefined);

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
    const restartRequired = restartRequested || healthQuery.data?.restart_required === true;

    const activeTrainingJobCount = jobs.filter(
        (job): job is SchemaTrainJob =>
            job.type === 'training' && (job.status === 'running' || job.status === 'pending')
    ).length;

    const refetchHealth = useCallback(async () => {
        const { data } = await healthQuery.refetch();
        const instanceId = data?.instance_id;
        if (instanceId === undefined) {
            return;
        }

        if (lastInstanceId.current !== undefined && instanceId !== lastInstanceId.current) {
            // The server has restarted, so clear the query cache. Queries may hold stale data from
            // before the restart, for example a project whose installed plugin was removed, or
            // plugin data that only becomes available again after the restart.
            queryClient.clear();
            setIsRestarting(false);
            setRestartRequested(false);
            setRestartPromptOpen(false);
        }
        lastInstanceId.current = instanceId;
    }, [healthQuery, queryClient]);

    const triggerRestartRequired = () => {
        setRestartRequested(true);
    };

    const openRestartPrompt = () => {
        setRestartPromptOpen(true);
    };

    const closeRestartPrompt = useCallback(() => {
        if (!isRestarting) {
            setRestartPromptOpen(false);
        }
    }, [isRestarting]);

    const restartServer = useCallback(async () => {
        if (isRestarting) {
            return;
        }

        // Record the current instance id so we can confirm the restart once it changes.
        await refetchHealth();

        setRestartPromptOpen(true);
        setIsRestarting(true);

        try {
            await restartMutation.mutateAsync({});
        } catch {
            // The backend may replace itself before the response is returned.
        }
    }, [isRestarting, refetchHealth, restartMutation]);

    useEffect(() => {
        if (!isRestarting || healthQuery.isFetching) {
            return;
        }

        const timeout = window.setTimeout(() => {
            void refetchHealth();
        }, HEALTH_POLL_INTERVAL_MS);

        return () => window.clearTimeout(timeout);
    }, [healthQuery, isRestarting, refetchHealth]);

    const value = useMemo(
        (): RestartStateValue => ({
            restartRequired,
            isRestarting,
            restartPromptOpen,
            activeTrainingJobCount,
            hasActiveTrainingJobs: activeTrainingJobCount > 0,
            triggerRestartRequired,
            openRestartPrompt,
            closeRestartPrompt,
            restartServer,
        }),
        [activeTrainingJobCount, closeRestartPrompt, isRestarting, restartPromptOpen, restartRequired, restartServer]
    );

    return (
        <RestartStateContext.Provider value={value}>
            {children}
            {restartRequired && restartPromptOpen ? (
                <RestartPromptDialog
                    isRestarting={isRestarting}
                    activeTrainingJobCount={activeTrainingJobCount}
                    onDismiss={closeRestartPrompt}
                    onRestart={restartServer}
                />
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
