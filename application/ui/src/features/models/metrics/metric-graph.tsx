import { useId } from 'react';

import { Flex, Text, View } from '@geti-ui/ui';
import {
    Area,
    AreaChart,
    CartesianGrid,
    ResponsiveContainer,
    Tooltip,
    TooltipContentProps,
    XAxis,
    YAxis,
} from 'recharts';

import { Box } from '../shared/box';
import type { MetricsEntry } from './types';

type CustomTooltipContentProps = Partial<TooltipContentProps> & {
    xAxisLabel?: string;
    yAxisLabel?: string;
    formatY?: (y: number) => string;
};

const CustomTooltipContent = ({
    active,
    payload,
    label,
    xAxisLabel,
    yAxisLabel,
    formatY,
}: CustomTooltipContentProps) => {
    if (!active || !payload?.length) {
        return null;
    }

    const value = payload[0].value;
    const formattedValue = formatY && value ? formatY(Number(value)) : value;

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
                {xAxisLabel}: {label}
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

const X_AXIS_TICK_COUNT = 8;
const Y_AXIS_TICK_COUNT = 4;
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
    const gradientId = useId();

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
                        <ResponsiveContainer width='100%' height={300} style={{ userSelect: 'none' }}>
                            <AreaChart
                                syncId={syncId}
                                style={{ aspectRatio: 1.6 }}
                                data={data}
                                margin={{ top: 35, bottom: 35, left: 35, right: 35 }}
                            >
                                <defs>
                                    <linearGradient id={gradientId} x1='0' y1='0' x2='0' y2='1'>
                                        <stop offset='5%' stopColor='var(--metric-graph-color)' stopOpacity={0.3} />
                                        <stop offset='95%' stopColor='var(--metric-graph-color)' stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid />
                                <XAxis
                                    dataKey={getX}
                                    type='number'
                                    domain={[0, 'dataMax']}
                                    label={{ value: xAxisLabel ?? 'x', position: 'bottom', fill: '#666', offset: 12 }}
                                    tickCount={X_AXIS_TICK_COUNT}
                                    tickMargin={12}
                                />
                                <YAxis
                                    label={{ value: yAxisLabel, angle: -90, position: 'center', dx: -38, fill: '#666' }}
                                    tickCount={Y_AXIS_TICK_COUNT}
                                    tickMargin={12}
                                    tickFormatter={(value) => formatYAxisTick(Number(value))}
                                />
                                <Area
                                    type='linear'
                                    dataKey={getY}
                                    name={yAxisLabel}
                                    stroke='var(--metric-graph-color)'
                                    strokeWidth={2}
                                    fill={`url(#${gradientId})`}
                                    dot={false}
                                    connectNulls
                                />
                                <Tooltip
                                    filterNull={false}
                                    content={
                                        <CustomTooltipContent
                                            xAxisLabel={xAxisLabel}
                                            yAxisLabel={yAxisLabel}
                                            formatY={formatY}
                                        />
                                    }
                                    cursor={{
                                        stroke: 'var(--metric-graph-color)',
                                    }}
                                />
                            </AreaChart>
                        </ResponsiveContainer>
                    </View>
                }
            />
        </Flex>
    );
};
