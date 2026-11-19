import { Text, View } from '@geti-ui/ui';
import { Tooltip, type TooltipContentProps } from 'recharts';

import type { MetricLine } from './types';

import classes from './metric-graph.module.css';

interface MetricTooltipProps {
    title: string;
    xAxisLabel?: string;
    formatY?: (y: number) => string;
    lines?: MetricLine[];
    highlightedLine?: string;
    onHighlight: (name: string | undefined) => void;
}

type MetricTooltipContentProps = MetricTooltipProps & Partial<TooltipContentProps>;

export const MetricTooltipContent = ({
    active,
    payload,
    label,
    title,
    xAxisLabel,
    formatY,
    lines,
    highlightedLine,
    onHighlight,
}: MetricTooltipContentProps) => {
    // Recharts only excludes hidden series when filterNull is enabled. Keep missing-value
    // rows useful while still respecting the legend's visibility controls.
    const visiblePayload = payload?.filter((item) => !item.hide);
    if (!active || !visiblePayload?.length) return null;

    return (
        <div role='tooltip' aria-label={`${title} values`} onMouseMove={(event) => event.stopPropagation()}>
            <View
                padding='size-100'
                backgroundColor='gray-50'
                borderRadius='regular'
                borderWidth='thin'
                borderColor='gray-400'
                UNSAFE_style={{
                    color: 'var(--spectrum-global-color-gray-900)',
                    fontSize: 'var(--spectrum-global-dimension-font-size-75)',
                }}
            >
                <div>
                    {xAxisLabel ?? 'Step'}: {label}
                </div>
                {visiblePayload.map(({ name, value, color }) => {
                    const line = lines?.find((candidate) => candidate.name === name);
                    const emphasized = line?.primary || highlightedLine === name;
                    const content = (
                        <>
                            {name}: {value == null ? 'Not available' : formatY ? formatY(Number(value)) : value}
                        </>
                    );

                    return lines ? (
                        <button
                            key={String(name)}
                            type='button'
                            className={classes.tooltipRow}
                            aria-label={`Highlight ${name}`}
                            onMouseEnter={() => onHighlight(String(name))}
                            onMouseLeave={() => onHighlight(undefined)}
                            onFocus={() => onHighlight(String(name))}
                            onBlur={() => onHighlight(undefined)}
                            style={{ color, fontWeight: emphasized ? 700 : 400 }}
                        >
                            {content}
                        </button>
                    ) : (
                        <Text key={String(name)} UNSAFE_style={{ color, display: 'block' }}>
                            {content}
                        </Text>
                    );
                })}
            </View>
        </div>
    );
};

export const MetricTooltip = (props: MetricTooltipProps) => (
    <Tooltip
        filterNull={false}
        includeHidden={false}
        itemSorter={undefined}
        wrapperStyle={props.lines ? { pointerEvents: 'auto', zIndex: 1 } : undefined}
        content={<MetricTooltipContent {...props} />}
        cursor={{ stroke: 'var(--metric-graph-color)' }}
    />
);
