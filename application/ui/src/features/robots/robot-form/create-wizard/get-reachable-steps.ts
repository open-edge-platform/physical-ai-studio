import type { RobotFamily } from './robot-families';

export type Step = 'robot' | 'variant' | 'configure';

/**
 * A step is reachable only once every step before it has been completed:
 * - `robot` is always reachable;
 * - `variant` needs a chosen family;
 * - `configure` needs a variant (robot type) that belongs to the chosen family.
 */
export const getReachableSteps = (family: RobotFamily | undefined, activeType: string | undefined): Step[] => {
    const reachable: Step[] = ['robot'];

    if (family === undefined) {
        return reachable;
    }

    reachable.push('variant');

    if (activeType !== undefined && family.variants.some(({ type }) => type === activeType)) {
        reachable.push('configure');
    }

    return reachable;
};
