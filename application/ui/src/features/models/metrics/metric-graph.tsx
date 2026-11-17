import { useCallback, useMemo, useState } from 'react';

import { Flex, Text, View } from '@geti-ui/ui';
import uPlot from 'uplot';

import { resolveCssColor, useUPlot, withAlpha } from '../../../components/uplot/use-uplot';
import { Box } from '../shared/box';
import type { MetricsEntry } from './types';

import classes from './metric-graph.module.css';

type CustomTooltipContentProps = {
    x: number;
    y: number | null | undefined;
    xAxisLabel?: string;
    yAxisLabel?: string;
    formatY?: (y: number) => string;
};

const CustomTooltipContent = ({ x, y, xAxisLabel, yAxisLabel, formatY }: CustomTooltipContentProps) => {
    const formattedValue = formatY && y ? formatY(Number(y)) : y;

    return (
        <View
            padding={'size-200'}
            backgroundColor={'gray-50'}
            borderRadius={'regular'}
            borderWidth={'thin'}
            borderColor={'gray-400'}
            UNSAFE_style={{
                padding: 'var(--spectrum-global-dimension-size-100)',
                color: 'var(--spectrum-global-color-gray-900)',
                fontSize: 'var(--spectrum-global-dimension-font-size-75)',
            }}
        >
            <div>
                {xAxisLabel}: {x}
            </div>
            <Text UNSAFE_style={{ color: 'var(--metric-graph-color)' }}>
                {yAxisLabel}: {formattedValue ?? 'Not available'}
            </Text>
        </View>
    );
};

type MetricGraphProps = {
    syncId?: string;
    title: string;
    data?: MetricsEntry[];
    xAxisLabel?: string;
    yAxisLabel: string;
    color?: string;
    getX: (metricsEntry: MetricsEntry) => number;
    getY: (metricsEntry: MetricsEntry) => number | null | undefined;
    formatY?: (y: number) => string;
};

const CHART_HEIGHT = 300;
// uPlot picks tick counts from the minimum pixel spacing between ticks; these approximate
// 8 ticks on the x axis and 4 on the y axis at the default chart size.
const X_AXIS_TICK_SPACE = 60;
const Y_AXIS_TICK_SPACE = 50;
const TOOLTIP_OFFSET = 12;
const Y_AXIS_SCIENTIFIC_THRESHOLD = 0.001;

const scientificFormatter = new Intl.NumberFormat('en-US', {
    notation: 'scientific',
    maximumSignificantDigits: 2,
    maximumFractionDigits: 2,
    roundingPriority: 'lessPrecision',
});

const fixedPointFormatter = new Intl.NumberFormat('en-US', {
    maximumFractionDigits: 4,
});

const formatYAxisTick = (value: number): string => {
    if (value !== 0 && Math.abs(value) < Y_AXIS_SCIENTIFIC_THRESHOLD) {
        return scientificFormatter.format(value);
    }

    return fixedPointFormatter.format(value);
};

interface HoveredPoint {
    index: number;
    left: number;
    top: number;
    /** Show the tooltip left of the cursor so it stays inside the chart on the right half. */
    flip: boolean;
}

export const MetricGraph = ({
    syncId,
    title,
    data,
    xAxisLabel,
    yAxisLabel,
    getY,
    getX,
    formatY,
    color = 'var(--energy-blue)',
}: MetricGraphProps) => {
    const [hovered, setHovered] = useState<HoveredPoint | null>(null);

    const chartData = useMemo<uPlot.AlignedData>(() => {
        const entries = data ?? [];

        return [entries.map(getX), entries.map((entry) => getY(entry) ?? null)];
    }, [data, getX, getY]);

    const buildOptions = useCallback(
        (container: HTMLDivElement): Omit<uPlot.Options, 'width' | 'height'> => {
            const stroke = resolveCssColor(container, color);
            const axisColor = resolveCssColor(container, 'var(--spectrum-global-color-gray-700)');
            const gridColor = resolveCssColor(container, 'var(--spectrum-global-color-gray-300)');

            const axis = {
                stroke: axisColor,
                grid: { stroke: gridColor, width: 1 },
                ticks: { stroke: gridColor, width: 1 },
            };

            return {
                padding: [24, 24, 0, 0],
                legend: { show: false },
                cursor: {
                    y: false,
                    points: { show: false },
                    drag: { x: false, y: false },
                    ...(syncId === undefined ? {} : { sync: { key: syncId } }),
                },
                scales: {
                    x: { time: false, range: (_plot, _min, max) => [0, max] },
                },
                axes: [
                    { ...axis, label: xAxisLabel ?? 'x', space: X_AXIS_TICK_SPACE },
                    {
                        ...axis,
                        label: yAxisLabel,
                        space: Y_AXIS_TICK_SPACE,
                        size: 70,
                        values: (_plot, splits) => splits.map(formatYAxisTick),
                    },
                ],
                series: [
                    {},
                    {
                        label: yAxisLabel,
                        stroke,
                        width: 2,
                        spanGaps: true,
                        points: { show: false },
                        fill: (plot) => {
                            const { top, height } = plot.bbox;
                            const gradient = plot.ctx.createLinearGradient(0, top, 0, top + height);
                            gradient.addColorStop(0.05, withAlpha(plot.ctx, stroke, 0.3));
                            gradient.addColorStop(0.95, withAlpha(plot.ctx, stroke, 0));
                            return gradient;
                        },
                    },
                ],
                hooks: {
                    setCursor: [
                        (plot) => {
                            const { idx, left, top } = plot.cursor;
                            setHovered(
                                idx == null || left == null || left < 0 || top == null
                                    ? null
                                    : {
                                          index: idx,
                                          left: plot.over.offsetLeft + left,
                                          top: plot.over.offsetTop + Math.max(0, top),
                                          flip: left > plot.over.clientWidth / 2,
                                      }
                            );
                        },
                    ],
                },
            };
        },
        [syncId, color, xAxisLabel, yAxisLabel]
    );

    const { containerRef } = useUPlot(buildOptions, chartData, CHART_HEIGHT);

    return (
        <Flex
            flex={1}
            direction={'column'}
            minWidth={'size-5000'}
            UNSAFE_style={{
                '--metric-graph-color': color,
            }}
        >
            <Box
                title={title}
                content={
                    <View backgroundColor={'gray-50'} minHeight={'size-3000'}>
                        <div ref={containerRef} className={classes.chart}>
                            {hovered !== null && hovered.index < chartData[0].length && (
                                <div
                                    className={classes.tooltip}
                                    style={{
                                        left: hovered.flip
                                            ? hovered.left - TOOLTIP_OFFSET
                                            : hovered.left + TOOLTIP_OFFSET,
                                        top: hovered.top + TOOLTIP_OFFSET,
                                        transform: hovered.flip ? 'translateX(-100%)' : undefined,
                                    }}
                                >
                                    <CustomTooltipContent
                                        x={chartData[0][hovered.index]}
                                        y={chartData[1][hovered.index]}
                                        xAxisLabel={xAxisLabel}
                                        yAxisLabel={yAxisLabel}
                                        formatY={formatY}
                                    />
                                </div>
                            )}
                        </div>
                    </View>
                }
            />
        </Flex>
    );
};
