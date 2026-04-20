
import { Flex, View } from '@geti/ui';
import { CartesianGrid, Text, Line, LineChart, Area, AreaChart, Tooltip, XAxis, YAxis, ResponsiveContainer } from 'recharts';
import { Box } from './box.component';

export type MetricGraphPoint = {
    x: number;
    y: number;
};

type MetricGraphProps = {
    title: string;
    data?: MetricGraphPoint[];
    xAxisLabel?: string;
    yAxisLabel: string;
};

const X_AXIS_TICK_COUNT = 8;
const Y_AXIS_TICK_COUNT = 4;

export const MetricGraph = ({ title, data, xAxisLabel, yAxisLabel }: MetricGraphProps) => {
    console.log(data);
    return (
        <Flex flex={1} direction={'column'} minWidth={'size-5000'}>
            <Box
                title={title}
                content={
                    <View backgroundColor={'gray-50'} minHeight={'size-3000'}>
                        <ResponsiveContainer width='100%' height={300} style={{ userSelect: 'none' }}>
                            <AreaChart
                                style={{ aspectRatio: 1.6 }}
                                data={data}
                                margin={{ top: 35, bottom: 35, left: 35 }}
                            >
                                <defs>
                                    <linearGradient id='areaGradient' x1='0' y1='0' x2='0' y2='1'>
                                        <stop offset='5%' stopColor='var(--energy-blue)' stopOpacity={0.3} />
                                        <stop offset='95%' stopColor='var(--energy-blue)' stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid />
                                <XAxis
                                    dataKey='x'
                                    type='number'
                                    domain={[0, 'auto']}
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
                                <Tooltip
                                    contentStyle={{ backgroundColor: '#fff', border: '1px solid #ccc' }}
                                    labelStyle={{ color: '#333' }}
                                />
                                <Area
                                    type='linear'
                                    dataKey='y'
                                    name={yAxisLabel}
                                    stroke='var(--energy-blue)'
                                    strokeWidth={2}
                                    fill='url(#areaGradient)'
                                    dot={false}
                                    isAnimationActive={false}
                                />
                            </AreaChart>
                        </ResponsiveContainer>
                    </View>
                }
            />
        </Flex>
    );
};
