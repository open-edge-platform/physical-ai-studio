import type { MouseHandlerDataParam, TickItem } from 'recharts';

/** Return the receiving chart's closest sample, breaking ties towards the earlier step. */
export const syncNearestStep = (
    ticks: ReadonlyArray<TickItem>,
    { activeLabel }: Pick<MouseHandlerDataParam, 'activeLabel'>
): number => {
    const step = activeLabel === undefined ? NaN : Number(activeLabel);
    if (!Number.isFinite(step)) return -1;

    let closestIndex = -1;
    let closestDistance = Infinity;
    let closestStep = Infinity;

    ticks.forEach((tick, index) => {
        const candidate = Number(tick.value);
        const distance = Math.abs(candidate - step);
        if (distance < closestDistance || (distance === closestDistance && candidate < closestStep)) {
            closestIndex = index;
            closestDistance = distance;
            closestStep = candidate;
        }
    });

    return closestIndex;
};
