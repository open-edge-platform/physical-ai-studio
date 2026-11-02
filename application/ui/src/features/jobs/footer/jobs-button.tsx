// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { ActionButton, DialogContainer, Icon } from '@geti-ui/ui';
import { Bell } from '@geti-ui/ui/icons';
import { useQueryClient } from '@tanstack/react-query';

import { LogsDialog } from '../../logs/logs-dialog';
import { JobsDialog, JobStatusTabKey } from '../jobs-dialog';
import { JOBS_QUERY_KEY } from '../use-job-cache';

type ViewState = { view: 'closed' } | { view: 'jobs' } | { view: 'logs'; jobId: string };

export const JobsButton = ({ projectId }: { projectId?: string }) => {
    const queryClient = useQueryClient();
    const [state, setState] = useState<ViewState>({ view: 'closed' });
    const [selectedTab, setSelectedTab] = useState<JobStatusTabKey>('all');

    const openFresh = () => {
        setSelectedTab('all');
        queryClient.invalidateQueries({ queryKey: JOBS_QUERY_KEY });
        setState({ view: 'jobs' });
    };

    return (
        <>
            <ActionButton
                isQuiet
                UNSAFE_style={{ paddingRight: 'var(--spectrum-global-dimension-size-100)' }}
                onPress={openFresh}
            >
                <Icon>
                    <Bell />
                </Icon>
                Jobs
            </ActionButton>
            <DialogContainer onDismiss={() => setState({ view: 'closed' })}>
                {state.view === 'jobs' && (
                    <JobsDialog
                        projectId={projectId}
                        selectedTab={selectedTab}
                        onSelectedTabChange={setSelectedTab}
                        onViewLogs={(job) => {
                            if (job.id === undefined) return;
                            setState({ view: 'logs', jobId: job.id });
                        }}
                        close={() => setState({ view: 'closed' })}
                    />
                )}
            </DialogContainer>
            <DialogContainer type='fullscreen' onDismiss={() => setState({ view: 'jobs' })}>
                {state.view === 'logs' && (
                    <LogsDialog close={() => setState({ view: 'jobs' })} initialSourceId={`job-${state.jobId}`} />
                )}
            </DialogContainer>
        </>
    );
};
