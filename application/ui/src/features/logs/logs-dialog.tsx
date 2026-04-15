// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense, useMemo, useState } from 'react';

import { Button, ButtonGroup, Content, Dialog, Divider, Flex, Heading, Loading, Text } from '@geti-ui/ui';
import { experimental_streamedQuery as streamedQuery, useQuery } from '@tanstack/react-query';

import { fetchClient } from '../../api/client';
import { fetchSSE } from '../../api/fetch-sse';
import { LogContent } from './log-content';
import type { LogEntry } from './log-types';
import { SourcesPicker } from './sources-picker';

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

        return query.data.filter((entry): entry is LogEntry => {
            return (
                entry !== null &&
                typeof entry === 'object' &&
                'record' in entry &&
                entry.record !== null &&
                typeof entry.record === 'object' &&
                'level' in entry.record &&
                'time' in entry.record &&
                'message' in entry.record
            );
        });
    }, [query.data]);

    return <LogContent logs={validLogs} isLoading={query.isLoading} />;
};

export const LogsDialog = ({ close, initialSourceId }: { close: () => void; initialSourceId?: string }) => {
    const [selectedSourceId, setSelectedSourceId] = useState<string>(initialSourceId ?? 'application');

    return (
        <Dialog onDismiss={close}>
            <Heading>
                <Flex alignItems='center' gap='size-300'>
                    <Text>Logs</Text>
                    <Suspense>
                        <SourcesPicker selectedSourceId={selectedSourceId} setSelectedSourceId={setSelectedSourceId} />
                    </Suspense>
                </Flex>
            </Heading>
            <Divider />
            <Content>
                <Suspense fallback={<Loading mode='inline' />}>
                    <LogStreamContent sourceId={selectedSourceId} />
                </Suspense>
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={close}>
                    Close
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};
