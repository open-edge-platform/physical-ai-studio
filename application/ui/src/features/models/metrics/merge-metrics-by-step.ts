// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { MetricsEntry } from './types';

type MergedFields = Required<Omit<MetricsEntry, 'step'>>;

const coalesce = <T>(current: T | null | undefined, incoming: T | null | undefined): T | null =>
    incoming ?? current ?? null;

const mergeFields = (current: MetricsEntry, incoming: MetricsEntry): MergedFields => ({
    epoch: coalesce(current.epoch, incoming.epoch),
    train_loss: coalesce(current.train_loss, incoming.train_loss),
    val_loss: coalesce(current.val_loss, incoming.val_loss),
    'lr-AdamW': coalesce(current['lr-AdamW'], incoming['lr-AdamW']),
});

export const mergeMetricsByStep = (entries: MetricsEntry[]): MetricsEntry[] => {
    const byStep = new Map<number, MetricsEntry>();

    for (const entry of entries) {
        const current = byStep.get(entry.step);

        byStep.set(entry.step, {
            step: entry.step,
            ...mergeFields(current ?? entry, entry),
        });
    }

    return byStep
        .values()
        .toArray()
        .toSorted((a, b) => a.step - b.step);
};
