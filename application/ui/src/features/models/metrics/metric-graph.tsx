import { useId } from 'react';

import { Flex, View } from '@geti-ui/ui';
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

import { Box } from '../shared/box';

export type MetricGraphPoint = {
    x: number;
    y: number;
};

type MetricGraphProps = {
    title: string;
    data?: MetricGraphPoint[];
    xAxisLabel?: string;
    yAxisLabel: string;
    color?: string;
};

const X_AXIS_TICK_COUNT = 8;
const Y_AXIS_TICK_COUNT = 4;

export const MetricGraph = ({
    title,
    data,
    xAxisLabel,
    yAxisLabel,
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
                                    dataKey='x'
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
                                    tickFormatter={(value) => Number(value).toFixed(4)}
                                />
                                <Area
                                    type='linear'
                                    dataKey='y'
                                    name={yAxisLabel}
                                    stroke='var(--metric-graph-color)'
                                    strokeWidth={2}
                                    fill={`url(#${gradientId})`}
                                    dot={false}
                                />
                                <Tooltip
                                    labelFormatter={(label) => `${xAxisLabel}: ${label}`}
                                    cursor={{
                                        stroke: 'var(--metric-graph-color)',
                                    }}
                                    contentStyle={{
                                        backgroundColor: 'var(--spectrum-global-color-gray-50)',
                                        borderColor: 'var(--color-border-2)',
                                        borderRadius: 'var(--spectrum-alias-border-radius-regular)',
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
