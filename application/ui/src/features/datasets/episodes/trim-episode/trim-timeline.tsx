import { useMemo, useRef, useState } from 'react';

import { Flex, View } from '@geti-ui/ui';

import { SchemaEpisode } from '../../../../api/openapi-spec';
import { Player } from '../use-player';
import { usePlayerTime } from '../use-player-time';
import { clamp, getFullTrimRange, TrimRange } from './trim-range';
import { getJointSeries, getMotion } from './trim-timeline-data';
import { DimmedRegion, HoverTooltip, Playhead, TrimHandle } from './trim-timeline-overlays';
import { Filmstrip, GripperTrack, JointsTrack, MotionTrack, Ruler, SeriesLegend } from './trim-timeline-tracks';
import { TrimToolbar } from './trim-toolbar';
import { DEFAULT_FILMSTRIP_CAMERA, useFilmstrip } from './use-filmstrip';
import { useTrimDrag } from './use-trim-drag';

import classes from './trim-timeline.module.css';

const FILMSTRIP_THUMBNAILS = 16;

interface TrimTimelineProps {
    datasetId: string;
    episode: SchemaEpisode;
    player: Player;
    filmstripCamera?: string;
}

export const TrimTimeline = ({
    datasetId,
    episode,
    player,
    filmstripCamera = DEFAULT_FILMSTRIP_CAMERA,
}: TrimTimelineProps) => {
    const { length, fps } = episode;

    const [range, setRange] = useState<TrimRange>(() => getFullTrimRange(length));
    const { time, seek } = usePlayerTime(player);
    const tracksRef = useRef<HTMLDivElement>(null);

    const series = useMemo(() => getJointSeries(episode), [episode]);
    const joints = useMemo(() => series.filter(({ isGripper }) => !isGripper), [series]);
    const grippers = useMemo(() => series.filter(({ isGripper }) => isGripper), [series]);
    const motion = useMemo(() => getMotion(series, length), [series, length]);
    const thumbnails = useFilmstrip(datasetId, episode, { camera: filmstripCamera, count: FILMSTRIP_THUMBNAILS });

    const drag = useTrimDrag({ trackRef: tracksRef, episode, player, range, setRange, seek });

    const hoverFrame = drag.hoverFrame === undefined ? undefined : clamp(drag.hoverFrame, 0, length - 1);

    return (
        <View
            flexShrink={0}
            marginTop='size-100'
            paddingTop='size-100'
            paddingX='size-125'
            paddingBottom='size-125'
            borderWidth='thin'
            borderColor='gray-200'
            borderRadius='regular'
            backgroundColor='gray-75'
        >
            <Flex direction='column' gap='size-75'>
                <TrimToolbar
                    datasetId={datasetId}
                    episode={episode}
                    player={player}
                    time={time}
                    seek={seek}
                    range={range}
                    setRange={setRange}
                />

                <Flex alignItems='center' wrap columnGap='size-175' rowGap='size-50' UNSAFE_className={classes.legend}>
                    <SeriesLegend series={series} />
                </Flex>

                <div
                    ref={tracksRef}
                    className={classes.tracks}
                    onPointerDown={drag.onPointerDown('playhead')}
                    onPointerMove={drag.onPointerMove}
                    onPointerUp={drag.onPointerUp}
                    onPointerLeave={drag.onPointerLeave}
                >
                    <Filmstrip thumbnails={thumbnails} />
                    <Ruler duration={length / fps} />
                    <JointsTrack joints={joints} />
                    {grippers.map((gripper) => (
                        <GripperTrack key={gripper.key} gripper={gripper} />
                    ))}
                    <MotionTrack motion={motion} />

                    <DimmedRegion from={0} to={range.start} length={length} />
                    <DimmedRegion from={range.end} to={length} length={length} />

                    {hoverFrame !== undefined && drag.dragging === undefined && (
                        <HoverTooltip frame={hoverFrame} length={length} fps={fps} series={series} />
                    )}

                    <Playhead position={time * fps} length={length} />
                    {(['start', 'end'] as const).map((edge) => (
                        <TrimHandle
                            key={edge}
                            frame={range[edge]}
                            length={length}
                            isActive={drag.dragging === edge}
                            onPointerDown={drag.onPointerDown(edge)}
                            onPointerMove={drag.onPointerMove}
                            onPointerUp={drag.onPointerUp}
                        />
                    ))}
                </div>
            </Flex>
        </View>
    );
};
