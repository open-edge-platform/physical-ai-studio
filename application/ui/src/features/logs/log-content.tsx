// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useMemo, useRef, useState } from 'react';

import { AriaListBox, AriaListBoxItem, Flex, ListLayout, Text, View, Virtualizer } from '@geti-ui/ui';

import { LogEntry } from './log-entry';
import { LogFilters } from './log-filters';
import { DEFAULT_LOG_FILTERS, type LogEntry as LogEntryType, type LogFilters as LogFiltersType } from './log-types';

import styles from './log-viewer.module.scss';

const formatLogForCopy = (log: LogEntryType): string => {
    const timestamp = new Date(log.record.time.timestamp * 1000).toISOString();
    const level = log.record.level.name.padEnd(8);
    const source = `${log.record.module}:${log.record.function}:${log.record.line}`;
    return `[${timestamp}] ${level} ${source} - ${log.record.message}`;
};

const getLogFilter = (filters: LogFiltersType) => {
    return (log: LogEntryType) => {
        // Filter by level
        if (!filters.levels.has(log.record.level.name)) {
            return false;
        }

        // Filter by search query
        if (filters.searchQuery) {
            const query = filters.searchQuery.toLowerCase();
            const message = log.record.message.toLowerCase();
            const module = log.record.module.toLowerCase();
            const func = log.record.function.toLowerCase();

            if (!message.includes(query) && !module.includes(query) && !func.includes(query)) {
                return false;
            }
        }

        // Filter by time range
        if (filters.startTime !== null && log.record.time.timestamp < filters.startTime) {
            return false;
        }
        if (filters.endTime !== null && log.record.time.timestamp > filters.endTime) {
            return false;
        }

        return true;
    };
};

const toVirtualizedLog = (entry: LogEntryType, idx: number) => {
    // eslint-disable-next-line max-len
    const id = `${entry.record.time.timestamp}-${entry.record.module}-${entry.record.function}-${entry.record.line}-${idx}`;

    return { id, entry };
};

const NoLogs = ({ isLoading, totalLogs }: { isLoading: boolean; totalLogs: number }) => {
    return (
        <Flex UNSAFE_className={styles.emptyState}>
            {isLoading ? (
                <Text>Loading logs...</Text>
            ) : totalLogs === 0 ? (
                <Text>No logs available</Text>
            ) : (
                <Text>No logs match the current filters</Text>
            )}
        </Flex>
    );
};

const useScrollToBottom = (totalLogs: number) => {
    const logsListRef = useRef<HTMLDivElement | null>(null);
    const [enabled, setEnabled] = useState(true);

    useEffect(() => {
        if (!enabled || !logsListRef.current) {
            return;
        }
        logsListRef.current.scrollTop = logsListRef.current.scrollHeight;
    }, [enabled, totalLogs]);

    return [enabled, setEnabled, logsListRef] as const;
};

const useFilteredLogs = (logs: Array<LogEntryType>, filters: LogFiltersType) => {
    return useMemo(() => {
        return logs.filter(getLogFilter(filters)).map(toVirtualizedLog);
    }, [logs, filters]);
};

export const LogContent = ({ logs, isLoading = false }: { logs: LogEntryType[]; isLoading?: boolean }) => {
    const [filters, setFilters] = useState<LogFiltersType>(DEFAULT_LOG_FILTERS);
    const filteredLogs = useFilteredLogs(logs, filters);
    const handleCopy = async () => {
        const formattedLogs = filteredLogs.map(({ entry }) => formatLogForCopy(entry)).join('\n');

        await navigator.clipboard.writeText(formattedLogs);
    };

    const [autoScroll, setAutoScroll, logsListRef] = useScrollToBottom(filteredLogs.length);

    return (
        <View UNSAFE_className={styles.logViewer}>
            <LogFilters
                filters={filters}
                onFiltersChange={setFilters}
                totalCount={logs.length}
                filteredCount={filteredLogs.length}
                autoScroll={autoScroll}
                onAutoScrollChange={setAutoScroll}
                handleCopy={handleCopy}
            />

            <div className={styles.logsContainer}>
                {filteredLogs.length === 0 ? (
                    <NoLogs isLoading={isLoading} totalLogs={logs.length} />
                ) : (
                    <View UNSAFE_className={styles.logsInner}>
                        <Virtualizer layout={ListLayout} layoutOptions={{ estimatedRowHeight: 36 }}>
                            <AriaListBox
                                aria-label='Log entries'
                                ref={logsListRef}
                                items={filteredLogs}
                                className={styles.virtualizedList}
                            >
                                {(item) => (
                                    <AriaListBoxItem id={item.id} textValue={item.entry.record.message}>
                                        <LogEntry entry={item.entry} />
                                    </AriaListBoxItem>
                                )}
                            </AriaListBox>
                        </Virtualizer>
                    </View>
                )}
            </div>
        </View>
    );
};
