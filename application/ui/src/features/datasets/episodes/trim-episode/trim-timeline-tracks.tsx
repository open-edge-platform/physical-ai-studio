import { memo, ReactNode, useMemo } from 'react';

import { Flex, Image, Text, View } from '@geti-ui/ui';

import { formatSeconds } from './trim-range';
import { getExtent, JointSeries, toPercent, toPolylinePoints, TRACK_HEIGHT, TRACK_WIDTH } from './trim-timeline-data';

import classes from './trim-timeline-tracks.module.css';

const JOINTS_TRACK_HEIGHT = 110;
const GRIPPER_TRACK_HEIGHT = 40;
const MOTION_TRACK_HEIGHT = 40;

interface SeriesSwatchProps {
    color: string;
}

export const SeriesSwatch = ({ color }: SeriesSwatchProps) => (
    <View
        width='size-125'
        height='size-40'
        flexShrink={0}
        borderRadius='small'
        UNSAFE_style={{ backgroundColor: color }}
    />
);

interface SeriesLegendProps {
    series: JointSeries[];
}

export const SeriesLegend = ({ series }: SeriesLegendProps) => (
    <>
        {series.map(({ key, label, color }) => (
            <Flex key={key} alignItems='center' gap='size-65' flexShrink={0} UNSAFE_className={classes.legendItem}>
                <SeriesSwatch color={color} />
                <Text>{label}</Text>
            </Flex>
        ))}
    </>
);

interface FilmstripProps {
    thumbnails: string[];
}

export const Filmstrip = ({ thumbnails }: FilmstripProps) => (
    <Flex height='size-600' UNSAFE_className={classes.filmstrip}>
        {thumbnails.map((src, index) => (
            <Image key={index} src={src} alt='' objectFit='cover' flex='1 1 0' minWidth={0} height='100%' />
        ))}
    </Flex>
);

interface RulerProps {
    duration: number;
}

export const Ruler = memo(({ duration }: RulerProps) => {
    const tickStep = duration > 60 ? 10 : duration > 20 ? 4 : 2;
    const ticks = Array.from({ length: Math.floor(duration / tickStep) + 1 }, (_, index) => index * tickStep);

    return (
        <View
            position='relative'
            height='size-200'
            borderBottomWidth='thin'
            borderBottomColor='gray-300'
            UNSAFE_className={classes.ruler}
        >
            {ticks.map((tick) => (
                <Text key={tick} UNSAFE_className={classes.tick} UNSAFE_style={{ left: toPercent(tick, duration) }}>
                    {formatSeconds(tick)}
                </Text>
            ))}
        </View>
    );
});

interface TrackProps {
    label: string;
    height: number;
    children: ReactNode;
}

const Track = ({ label, height, children }: TrackProps) => (
    <View position='relative' height={height} borderBottomWidth='thin' borderBottomColor='gray-200'>
        <Text UNSAFE_className={classes.trackLabel}>{label}</Text>
        <svg className={classes.trackSvg} viewBox={`0 0 ${TRACK_WIDTH} ${TRACK_HEIGHT}`} preserveAspectRatio='none'>
            {children}
        </svg>
    </View>
);

interface SeriesLineProps {
    series: JointSeries;
    extent: [min: number, max: number];
}

const SeriesLine = ({ series, extent }: SeriesLineProps) => (
    <polyline
        points={toPolylinePoints(series.values, extent)}
        fill='none'
        stroke={series.color}
        strokeWidth={1.5}
        vectorEffect='non-scaling-stroke'
    />
);

interface JointsTrackProps {
    joints: JointSeries[];
}

export const JointsTrack = memo(({ joints }: JointsTrackProps) => {
    const extent = useMemo(() => getExtent(joints.flatMap(({ values }) => values)), [joints]);

    return (
        <Track
            label={`joint positions · ${extent[0].toFixed(0)} … ${extent[1].toFixed(0)}`}
            height={JOINTS_TRACK_HEIGHT}
        >
            {joints.map((joint) => (
                <SeriesLine key={joint.key} series={joint} extent={extent} />
            ))}
        </Track>
    );
});

interface GripperTrackProps {
    gripper: JointSeries;
}

export const GripperTrack = memo(({ gripper }: GripperTrackProps) => {
    const extent = useMemo(() => getExtent(gripper.values), [gripper]);

    return (
        <Track label={gripper.label} height={GRIPPER_TRACK_HEIGHT}>
            <SeriesLine series={gripper} extent={extent} />
        </Track>
    );
});

interface MotionTrackProps {
    motion: number[];
}

export const MotionTrack = memo(({ motion }: MotionTrackProps) => {
    const points = useMemo(() => toPolylinePoints(motion, getExtent(motion)), [motion]);

    return (
        <Track label='motion — Σ|Δjoint| per frame, near zero means the arm is idle' height={MOTION_TRACK_HEIGHT}>
            <polygon points={`0,${TRACK_HEIGHT} ${points} ${TRACK_WIDTH},${TRACK_HEIGHT}`} className={classes.motion} />
        </Track>
    );
});
