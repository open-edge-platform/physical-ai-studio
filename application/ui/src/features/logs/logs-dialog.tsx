// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense, useMemo, useState } from 'react';

import {
    Button,
    ButtonGroup,
    Content,
    Dialog,
    Divider,
    Flex,
    Heading,
    IllustratedMessage,
    Loading,
    Text,
} from '@geti-ui/ui';
import { AlertCircle } from '@geti-ui/ui/icons';
import { experimental_streamedQuery as streamedQuery, useQuery } from '@tanstack/react-query';

import { fetchClient } from '../../api/client';
import { fetchSSE } from '../../api/fetch-sse';
import { ErrorBoundary } from '../../components/error-boundary/error-boundary';
import { LogContent } from './log-content';
import type { LogEntry } from './log-types';
import { SourcesPicker } from './sources-picker';

/** Narrow an unknown streamed SSE payload down to a well-formed `LogEntry`.
 *
 * Log sources for jobs that just started (in particular remote SSH jobs, whose
 * log file may not exist or may only contain a partial first line yet) can
 * surface unexpected payload shapes. Checking only that the keys exist -- not
 * that their values have the right *type* -- lets a malformed entry through,
 * which then throws deep in rendering (e.g. `message.includes(...)` on a
 * non-string). Since nothing in the tree catches that, it used to bubble all
 * the way up to the router's root `errorElement` and blank the whole app. Drop
 * anything that doesn't fully match instead.
 */
const isValidLogEntry = (entry: unknown): entry is LogEntry => {
    if (entry === null || typeof entry !== 'object' || !('record' in entry)) {
        return false;
    }

    const record = (entry as { record: unknown }).record;
    if (record === null || typeof record !== 'object') {
        return false;
    }

    const { level, time, message, module, function: fn, line } = record as Record<string, unknown>;

    return (
        typeof message === 'string' &&
        typeof module === 'string' &&
        typeof fn === 'string' &&
        typeof line === 'number' &&
        typeof level === 'object' &&
        level !== null &&
        typeof (level as Record<string, unknown>).name === 'string' &&
        typeof time === 'object' &&
        time !== null &&
        typeof (time as Record<string, unknown>).timestamp === 'number' &&
        typeof (time as Record<string, unknown>).repr === 'string'
    );
};

const LogStreamContent = ({ sourceId }: { sourceId: string }) => {
    const query = useQuery({
        queryKey: ['get', '/api/logs/{source_id}/stream', sourceId],
        queryFn: streamedQuery({
            streamFn: (context) => {
                const url = fetchClient.PATH('/api/logs/{source_id}/stream', {
                    params: { path: { source_id: sourceId } },
                });

                return fetchSSE<LogEntry>(url, { signal: context.signal });
            },
        }),
        staleTime: Infinity,
    });

    const validLogs = useMemo(() => {
        if (!query.data) return [];

        return query.data.filter(isValidLogEntry);
    }, [query.data]);

    return <LogContent logs={validLogs} isLoading={query.isLoading} />;
};

const LogsErrorFallback = ({ retry }: { retry: () => void }) => (
    <IllustratedMessage>
        <AlertCircle />
        <Heading>Something went wrong while displaying these logs</Heading>
        <Content>
            <Button variant='accent' onPress={retry}>
                Try again
            </Button>
        </Content>
    </IllustratedMessage>
);

export const LogsDialog = ({ close, initialSourceId }: { close: () => void; initialSourceId?: string }) => {
    const [selectedSourceId, setSelectedSourceId] = useState<string>(initialSourceId ?? 'application');

    return (
        <Dialog onDismiss={close}>
            <Heading>
                <Flex alignItems='center' gap='size-300'>
                    <Text>Logs</Text>
                    <ErrorBoundary fallback={() => null}>
                        <Suspense>
                            <SourcesPicker
                                selectedSourceId={selectedSourceId}
                                setSelectedSourceId={setSelectedSourceId}
                            />
                        </Suspense>
                    </ErrorBoundary>
                </Flex>
            </Heading>
            <Divider />
            <Content>
                <ErrorBoundary fallback={(retry) => <LogsErrorFallback retry={retry} />}>
                    <Suspense fallback={<Loading mode='inline' />}>
                        <LogStreamContent sourceId={selectedSourceId} />
                    </Suspense>
                </ErrorBoundary>
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={close}>
                    Close
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};
