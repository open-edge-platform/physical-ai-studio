// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { MetricsEntry } from './types';

const coalesce = <T>(current: T | null | undefined, incoming: T | null | undefined): T | null =>
    incoming ?? current ?? null;

const mergeEntries = (current: MetricsEntry, incoming: MetricsEntry): MetricsEntry => {
    const merged: MetricsEntry = {
        step: incoming.step,
        epoch: null,
        train_loss: null,
        val_loss: null,
        'lr-AdamW': null,
    };
    for (const key of new Set([...Object.keys(current), ...Object.keys(incoming)])) {
        if (key !== 'step') {
            merged[key] = coalesce(current[key], incoming[key]);
        }
    }
    return merged;
};

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

        next.set(entry.step, mergeEntries(current ?? entry, entry));
    }

    return next;
};

/** Converts a step -> entry map into an array sorted by step, ascending. */
export const sortMetricsByStep = (byStep: ReadonlyMap<number, MetricsEntry>): MetricsEntry[] =>
    byStep
        .values()
        .toArray()
        .toSorted((a, b) => a.step - b.step);
