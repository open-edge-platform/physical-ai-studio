import { useRef } from 'react';

import { useContainerSize } from '../../components/zoom/use-container-size';
import { calculateMediaFit } from './media-fit';

export const useFittedMediaSize = (intrinsicWidth?: number, intrinsicHeight?: number) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const containerSize = useContainerSize(containerRef);

    const w = Number(intrinsicWidth);
    const h = Number(intrinsicHeight);
    const aspectRatio = w > 0 && Number.isFinite(w) && h > 0 && Number.isFinite(h) ? w / h : 1;

    const { width, height } = calculateMediaFit(containerSize, aspectRatio);

    return { containerRef, width, height };
};
