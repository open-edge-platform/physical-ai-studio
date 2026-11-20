import { useId, useState } from 'react';

import { Content, ContextualHelp, Flex, Heading, Text, View } from '@geti-ui/ui';
import {
    Area,
    CartesianGrid,
    ComposedChart,
    Curve,
    Line,
    LineDrawShape,
    ResponsiveContainer,
    XAxis,
    YAxis,
    type LineDrawShapeProps,
} from 'recharts';

import { Box } from '../shared/box';
import { MetricTooltip } from './metric-tooltip';
import { syncNearestStep } from './sync-nearest-step';
import { hasMetricValue, type MetricLine, type MetricsEntry } from './types';

import classes from './metric-graph.module.css';

const HighlightableLine = ({
    onHighlight,
    ...props
}: LineDrawShapeProps & { onHighlight: (active: boolean) => void }) => (
    <g>
        <LineDrawShape {...props} />
        <Curve
            points={props.points}
            type='linear'
            connectNulls
            stroke='transparent'
            strokeWidth={14}
            fill='none'
            pointerEvents='stroke'
            onMouseEnter={() => onHighlight(true)}
            onMouseLeave={() => onHighlight(false)}
        />
    </g>
);

type MetricGraphProps = {
    syncId?: string;
    title: string;
    helpText?: string;
    data?: MetricsEntry[];
    xAxisLabel?: string;
    yAxisLabel: string;
    color?: string;
    getX: (metricsEntry: MetricsEntry) => number;
    getY: (metricsEntry: MetricsEntry) => number | null | undefined;
    // Draws one line per entry, with a legend, instead of the single `getY` area.
    lines?: MetricLine[];
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
    helpText,
    data,
    xAxisLabel,
    yAxisLabel,
    getY,
    getX,
    lines,
    formatY,
    color = 'var(--energy-blue)',
}: MetricGraphProps) => {
    const gradientId = useId();
    const [hiddenLines, setHiddenLines] = useState<Set<string>>(new Set());
    const [highlightedLine, setHighlightedLine] = useState<string>();
    const visibleLines = lines?.filter((line) => !hiddenLines.has(line.name));
    const hasPrimaryLine = lines?.some((line) => line.primary);
    // Train and validation rows are logged at different steps; keep only this chart's rows so the
    // tooltip snaps to a point that has values, and sync the charts by step instead of row index.
    const points = data?.filter((entry) => hasMetricValue(entry, getY, visibleLines?.length ? visibleLines : lines));

    const toggleLine = (name: string) => {
        setHiddenLines((current) => {
            const next = new Set(current);
            if (next.has(name)) next.delete(name);
            else next.add(name);
            return next;
        });
        setHighlightedLine(undefined);
    };

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
                title={
                    <span className={classes.heading}>
                        <span>{title}</span>
                        {helpText && (
                            <ContextualHelp variant='info' aria-label={`${title} information`}>
                                <Heading>{title}</Heading>
                                <Content>
                                    <Text>{helpText}</Text>
                                </Content>
                            </ContextualHelp>
                        )}
                    </span>
                }
                contentClassName={classes.content}
                content={
                    <View backgroundColor={'gray-50'} UNSAFE_className={classes.content}>
                        <section aria-label={`${title} graph`} className={classes.content}>
                            {lines && (
                                <div className={classes.legend} aria-label={`${title} series`}>
                                    {lines.map((line) => (
                                        <button
                                            key={line.name}
                                            type='button'
                                            className={classes.legendButton}
                                            aria-pressed={!hiddenLines.has(line.name)}
                                            onClick={() => toggleLine(line.name)}
                                            onMouseEnter={() => setHighlightedLine(line.name)}
                                            onMouseLeave={() => setHighlightedLine(undefined)}
                                            onFocus={() => setHighlightedLine(line.name)}
                                            onBlur={() => setHighlightedLine(undefined)}
                                            style={{ fontWeight: line.primary ? 700 : 400 }}
                                        >
                                            <span
                                                aria-hidden
                                                className={classes.swatch}
                                                style={{
                                                    borderColor: line.color,
                                                    borderTopStyle: line.strokeDasharray ? 'dashed' : 'solid',
                                                }}
                                            />
                                            {line.name}
                                        </button>
                                    ))}
                                </div>
                            )}
                            <ResponsiveContainer
                                width='100%'
                                height='100%'
                                minHeight={300}
                                className={classes.chart}
                                style={{ userSelect: 'none' }}
                            >
                                <ComposedChart
                                    syncId={syncId}
                                    syncMethod={syncNearestStep}
                                    style={{ aspectRatio: 1.6 }}
                                    data={points}
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
                                        label={{
                                            value: xAxisLabel ?? 'x',
                                            position: 'bottom',
                                            fill: '#666',
                                            offset: 12,
                                        }}
                                        tickCount={X_AXIS_TICK_COUNT}
                                        tickMargin={12}
                                    />
                                    <YAxis
                                        label={{
                                            value: yAxisLabel,
                                            angle: -90,
                                            position: 'center',
                                            dx: -38,
                                            fill: '#666',
                                        }}
                                        tickCount={Y_AXIS_TICK_COUNT}
                                        tickMargin={12}
                                        tickFormatter={(value) => formatYAxisTick(Number(value))}
                                    />
                                    {lines ? (
                                        lines.map((line) => (
                                            <Line
                                                key={line.name}
                                                type='linear'
                                                dataKey={line.getY}
                                                name={line.name}
                                                stroke={line.color}
                                                strokeWidth={line.strokeWidth ?? (hasPrimaryLine ? 1.5 : 2)}
                                                strokeOpacity={
                                                    hasPrimaryLine && !line.primary && highlightedLine !== line.name
                                                        ? 0.25
                                                        : 1
                                                }
                                                strokeDasharray={line.strokeDasharray}
                                                hide={hiddenLines.has(line.name)}
                                                zIndex={highlightedLine === line.name ? 600 : line.primary ? 500 : 400}
                                                isAnimationActive={false}
                                                shape={(props: LineDrawShapeProps) => (
                                                    <HighlightableLine
                                                        {...props}
                                                        onHighlight={(active) =>
                                                            setHighlightedLine(active ? line.name : undefined)
                                                        }
                                                    />
                                                )}
                                                dot={false}
                                                connectNulls
                                            />
                                        ))
                                    ) : (
                                        <Area
                                            type='linear'
                                            isAnimationActive={false}
                                            dataKey={getY}
                                            name={yAxisLabel}
                                            stroke='var(--metric-graph-color)'
                                            strokeWidth={2}
                                            fill={`url(#${gradientId})`}
                                            dot={false}
                                            connectNulls
                                        />
                                    )}
                                    <MetricTooltip
                                        title={title}
                                        data={points}
                                        getX={getX}
                                        xAxisLabel={xAxisLabel}
                                        formatY={formatY}
                                        lines={visibleLines}
                                        highlightedLine={highlightedLine}
                                        onHighlight={setHighlightedLine}
                                    />
                                </ComposedChart>
                            </ResponsiveContainer>
                        </section>
                    </View>
                }
            />
        </Flex>
    );
};
