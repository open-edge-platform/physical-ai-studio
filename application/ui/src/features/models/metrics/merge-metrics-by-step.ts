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

/**
 * Folds a batch of raw metric entries into an existing step -> entry map,
 * without reprocessing entries that were already folded in on a previous
 * call. Always returns a new `Map` (never mutates `byStep`) so that callers
 * relying on referential equality to detect changes (e.g. TanStack Query)
 * observe the update.
 */
export const foldMetricsBatch = (
    byStep: ReadonlyMap<number, MetricsEntry>,
    batch: readonly MetricsEntry[]
): Map<number, MetricsEntry> => {
    const next = new Map(byStep);

    for (const entry of batch) {
        const current = next.get(entry.step);

        next.set(entry.step, {
            step: entry.step,
            ...mergeFields(current ?? entry, entry),
        });
    }

    return next;
};

/** Converts a step -> entry map into an array sorted by step, ascending. */
export const sortMetricsByStep = (byStep: ReadonlyMap<number, MetricsEntry>): MetricsEntry[] =>
    byStep
        .values()
        .toArray()
        .toSorted((a, b) => a.step - b.step);
