import { useEffect, useRef, useState } from 'react';

import { Flex, ProgressCircle } from '@geti-ui/ui';

import { InlineAlert } from '../robots/setup-wizard/shared/inline-alert';
import { useFittedMediaSize } from './use-fitted-media-size';

/** Where a camera view gets its JPEG frames from: a camera socket or a runtime session. */
export interface FrameSource {
    subscribe: (listener: (jpeg: Blob) => void) => () => void;
}

const CameraCanvas = ({
    source,
    label,
    width,
    height,
}: {
    source: FrameSource;
    label: string;
    width: number;
    height: number;
}) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        // Draw the newest frame only; frames that arrive while one decodes replace each other.
        let next: Blob | null = null;
        let drawing = false;
        let disposed = false;

        const draw = async () => {
            drawing = true;
            while (next !== null && !disposed) {
                const blob: Blob = next;
                next = null;
                try {
                    const bitmap = await createImageBitmap(blob);
                    const canvas = canvasRef.current;
                    canvas?.getContext('2d', { alpha: false })?.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
                    bitmap.close();
                    setIsLoading(false);
                } catch (error) {
                    console.error('Failed to draw camera frame:', error);
                }
            }
            drawing = false;
        };

        const unsubscribe = source.subscribe((jpeg) => {
            next = jpeg;
            if (!drawing) {
                void draw();
            }
        });
        return () => {
            disposed = true;
            unsubscribe();
        };
    }, [source]);

    return (
        <>
            {isLoading && (
                <Flex width='100%' height='100%' justifyContent='center' alignItems='center'>
                    <ProgressCircle isIndeterminate />
                </Flex>
            )}
            <canvas
                ref={canvasRef}
                width={width}
                height={height}
                style={{ display: isLoading ? 'none' : 'block' }}
                aria-label={label}
            />
        </>
    );
};

/** Draws frames from ``source``, fitted to its container at the camera's aspect ratio. */
export const CameraView = ({
    source,
    label,
    width,
    height,
    error,
}: {
    source: FrameSource;
    label: string;
    width?: number | null;
    height?: number | null;
    error?: string | null;
}) => {
    const fitted = useFittedMediaSize(Number(width), Number(height));

    if (error) {
        return (
            <Flex width='100%' height='100%' justifyContent='center' alignItems='center'>
                <InlineAlert variant='error'>{error}</InlineAlert>
            </Flex>
        );
    }

    return (
        <div ref={fitted.containerRef} style={{ height: '100%', width: '100%' }}>
            <CameraCanvas source={source} label={label} width={fitted.width} height={fitted.height} />
        </div>
    );
};
