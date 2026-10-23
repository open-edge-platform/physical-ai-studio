import { HTMLAttributes } from 'react';

import { Grid, Text, View } from '@geti-ui/ui';

import { formatSeconds } from './trim-range';
import { JointSeries, toPercent } from './trim-timeline-data';
import { SeriesSwatch } from './trim-timeline-tracks';

import classes from './trim-timeline-overlays.module.css';

const TOOLTIP_FLIP_THRESHOLD = 0.7;

const jointValueFormatter = new Intl.NumberFormat('en-US', { maximumFractionDigits: 3, useGrouping: false });

interface DimmedRegionProps {
    from: number;
    to: number;
    length: number;
}

export const DimmedRegion = ({ from, to, length }: DimmedRegionProps) => (
    <View
        position='absolute'
        top={0}
        bottom={0}
        left={toPercent(from, length)}
        width={toPercent(to - from, length)}
        UNSAFE_className={classes.dimmed}
    />
);

interface PlayheadProps {
    position: number;
    length: number;
}

export const Playhead = ({ position, length }: PlayheadProps) => (
    <View
        position='absolute'
        top={0}
        bottom={0}
        left={toPercent(position, length)}
        UNSAFE_className={classes.playhead}
    />
);

interface TrimHandleProps extends Pick<
    HTMLAttributes<HTMLDivElement>,
    'onPointerDown' | 'onPointerMove' | 'onPointerUp'
> {
    frame: number;
    length: number;
    isActive: boolean;
}

export const TrimHandle = ({ frame, length, isActive, ...pointerHandlers }: TrimHandleProps) => (
    <div
        className={classes.handle}
        data-active={isActive}
        style={{ left: toPercent(frame, length) }}
        {...pointerHandlers}
    >
        <View UNSAFE_className={classes.knob} />
    </div>
);

interface HoverTooltipProps {
    frame: number;
    length: number;
    fps: number;
    series: JointSeries[];
}

export const HoverTooltip = ({ frame, length, fps, series }: HoverTooltipProps) => {
    const isNearEnd = frame > length * TOOLTIP_FLIP_THRESHOLD;

    return (
        <View
            position='absolute'
            top={0}
            bottom={0}
            left={toPercent(frame, length)}
            UNSAFE_className={classes.hoverLine}
        >
            <View
                position='absolute'
                bottom='size-100'
                left={isNearEnd ? undefined : 'size-100'}
                right={isNearEnd ? 'size-100' : undefined}
                zIndex={3}
                minWidth='size-2400'
                paddingX='size-100'
                paddingY='size-75'
                borderWidth='thin'
                borderColor='gray-300'
                borderRadius='regular'
                UNSAFE_className={classes.tooltip}
            >
                <Text UNSAFE_className={classes.mono}>
                    frame {frame} · {formatSeconds(frame / fps)}
                </Text>
                {series.map(({ key, label, color, values }) => (
                    <Grid key={key} columns={['size-175', '1fr', 'auto']} alignItems='center' gap='size-50'>
                        <SeriesSwatch color={color} />
                        <Text>{label}</Text>
                        <Text UNSAFE_className={classes.mono}>{jointValueFormatter.format(values[frame])}</Text>
                    </Grid>
                ))}
            </View>
        </View>
    );
};
