import { PointerEvent, RefObject, useState } from 'react';

import { SchemaEpisode } from '../../../../api/openapi-spec';
import { Player } from '../use-player';
import { clamp, TrimRange, withTrimEnd, withTrimStart } from './trim-range';

export type DragTarget = 'start' | 'end' | 'playhead';

interface UseTrimDragProps {
    trackRef: RefObject<HTMLElement | null>;
    episode: SchemaEpisode;
    player: Player;
    range: TrimRange;
    setRange: (range: TrimRange) => void;
    seek: (time: number) => void;
}

export const useTrimDrag = ({ trackRef, episode, player, range, setRange, seek }: UseTrimDragProps) => {
    const [dragging, setDragging] = useState<DragTarget | undefined>(undefined);
    const [hoverFrame, setHoverFrame] = useState<number | undefined>(undefined);
    const { length, fps } = episode;

    const frameFromEvent = ({ clientX }: PointerEvent) => {
        const rect = trackRef.current?.getBoundingClientRect();
        if (rect === undefined || rect.width === 0) return 0;
        return clamp(Math.round(((clientX - rect.left) / rect.width) * length), 0, length);
    };

    const apply = (target: DragTarget, frame: number) => {
        if (target === 'start') {
            const next = withTrimStart(range, frame);
            setRange(next);
            seek(next.start / fps);
        } else if (target === 'end') {
            const next = withTrimEnd(range, frame, length);
            setRange(next);
            seek((next.end - 1) / fps);
        } else {
            seek(clamp(frame, 0, length - 1) / fps);
        }
    };

    const onPointerDown = (target: DragTarget) => (event: PointerEvent<HTMLElement>) => {
        event.stopPropagation();
        event.preventDefault();
        event.currentTarget.setPointerCapture(event.pointerId);
        player.pause();
        setDragging(target);
        if (target === 'playhead') apply(target, frameFromEvent(event));
    };

    const onPointerMove = (event: PointerEvent<HTMLElement>) => {
        const frame = frameFromEvent(event);
        setHoverFrame(frame);
        if (dragging !== undefined) apply(dragging, frame);
    };

    const onPointerUp = (event: PointerEvent<HTMLElement>) => {
        if (event.currentTarget.hasPointerCapture(event.pointerId)) {
            event.currentTarget.releasePointerCapture(event.pointerId);
        }
        setDragging(undefined);
    };

    const onPointerLeave = () => {
        if (dragging === undefined) setHoverFrame(undefined);
    };

    return { dragging, hoverFrame, onPointerDown, onPointerMove, onPointerUp, onPointerLeave };
};
