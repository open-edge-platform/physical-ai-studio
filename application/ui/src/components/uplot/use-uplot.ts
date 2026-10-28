import { useEffect, useRef } from 'react';

import uPlot from 'uplot';

import 'uplot/dist/uPlot.min.css';

/**
 * Resolves a CSS color (including `var(--token)` references) to a value canvas can draw with.
 *
 * Canvas does not understand custom properties, so the value is assigned to a scratch custom
 * property on `element` and read back computed, which substitutes any `var()` references.
 */
export const resolveCssColor = (element: HTMLElement, value: string): string => {
    element.style.setProperty('--uplot-resolve', value);
    const resolved = getComputedStyle(element).getPropertyValue('--uplot-resolve').trim();
    element.style.removeProperty('--uplot-resolve');

    return resolved || value;
};

/** Returns `color` (any canvas-parsable color) as `rgba(...)` with the given `alpha`. */
export const withAlpha = (ctx: CanvasRenderingContext2D, color: string, alpha: number): string => {
    // Assigning to fillStyle normalizes the color to `#rrggbb` (opaque) or `rgba(r, g, b, a)`.
    ctx.fillStyle = color;
    const normalized = String(ctx.fillStyle);

    if (normalized.startsWith('#')) {
        const [r, g, b] = [1, 3, 5].map((offset) => parseInt(normalized.slice(offset, offset + 2), 16));
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }

    return normalized.replace(/rgba?\(([^,]+),([^,]+),([^,)]+).*\)/, `rgba($1,$2,$3, ${alpha})`);
};

type BuildOptions = (container: HTMLDivElement) => Omit<uPlot.Options, 'width' | 'height'>;

/**
 * Mounts a uPlot chart into the returned `containerRef`, sized to the container's width and `height`.
 *
 * The chart is rebuilt when `buildOptions` changes (memoize it), while `data` changes only call
 * `setData` so streaming updates do not flash.
 */
export const useUPlot = (buildOptions: BuildOptions, data: uPlot.AlignedData, height: number) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const plotRef = useRef<uPlot | null>(null);
    const dataRef = useRef(data);
    dataRef.current = data;

    useEffect(() => {
        const container = containerRef.current;
        if (container === null) return;

        const plot = new uPlot(
            { ...buildOptions(container), width: container.clientWidth, height },
            dataRef.current,
            container
        );
        plotRef.current = plot;

        const observer = new ResizeObserver(() => {
            plot.setSize({ width: container.clientWidth, height });
        });
        observer.observe(container);

        return () => {
            observer.disconnect();
            plot.destroy();
            plotRef.current = null;
        };
    }, [buildOptions, height]);

    useEffect(() => {
        plotRef.current?.setData(data);
    }, [data]);

    return { containerRef, plotRef };
};
