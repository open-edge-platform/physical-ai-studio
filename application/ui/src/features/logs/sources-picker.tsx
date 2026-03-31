// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, SetStateAction, useMemo } from 'react';

import { Item, Picker, Section } from '@geti-ui/ui';
import { orderBy } from 'lodash-es';

import { $api } from '../../api/client';
import type { LogSource } from './log-types';

/** Format an ISO-8601 timestamp into a short locale string (date + time). */
const formatCreatedAt = (date: string | null | undefined): string | null => {
    if (!date) {
        return null;
    }

    try {
        return new Date(date).toLocaleString(undefined, {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
        });
    } catch {
        return null;
    }
};

const formatSourceLabel = (source: LogSource): string => {
    const createdAt = formatCreatedAt(source.created_at);
    if (createdAt === null) {
        return source.name;
    }

    return `${source.name} — ${createdAt}`;
};

export const SourcesPicker = ({
    selectedSourceId,
    setSelectedSourceId,
}: {
    selectedSourceId: string;
    setSelectedSourceId: Dispatch<SetStateAction<string>>;
}) => {
    const { data: sources, isLoading: sourcesLoading } = $api.useSuspenseQuery('get', '/api/logs/sources');

    // Build picker items — Spectrum Picker requires a flat array of Item/Section children.
    // We only render sections that have items.
    const pickerSections = useMemo(() => {
        const sections: { title: string; items: LogSource[] }[] = [];

        const applicationSources = sources.filter((s) => s.type === 'application');
        if (applicationSources.length > 0) {
            sections.push({ title: 'Application', items: applicationSources });
        }

        const workerSources = sources.filter((s) => s.type === 'worker');
        if (workerSources.length > 0) {
            sections.push({ title: 'Workers', items: workerSources });
        }

        const jobSources = orderBy(
            sources.filter((s) => s.type === 'job'),
            ['created_at'],
            ['desc']
        );
        if (jobSources.length > 0) {
            sections.push({ title: 'Jobs', items: jobSources });
        }

        return sections;
    }, [sources]);

    return (
        <Picker
            aria-label='Log source'
            selectedKey={selectedSourceId}
            onSelectionChange={(key) => setSelectedSourceId(String(key))}
            isDisabled={sourcesLoading}
            width='size-3600'
        >
            {pickerSections.map((section) => (
                <Section key={section.title} title={section.title}>
                    {section.items.map((s) => (
                        <Item key={s.id} textValue={formatSourceLabel(s)}>
                            {formatSourceLabel(s)}
                        </Item>
                    ))}
                </Section>
            ))}
        </Picker>
    );
};
