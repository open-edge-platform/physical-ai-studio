export type ContainerSize = { width: number; height: number };

export const calculateMediaFit = (
    containerSize: ContainerSize,
    aspectRatio: number
): { width: number; height: number } => {
    if (
        containerSize.width <= 0 ||
        !Number.isFinite(containerSize.width) ||
        containerSize.height <= 0 ||
        !Number.isFinite(containerSize.height)
    ) {
        return { width: 0, height: 0 };
    }

    const safeAspectRatio = aspectRatio > 0 && Number.isFinite(aspectRatio) ? aspectRatio : 1;

    const containerAspectRatio = containerSize.width / containerSize.height;

    if (containerAspectRatio > safeAspectRatio) {
        return {
            width: containerSize.height * safeAspectRatio,
            height: containerSize.height,
        };
    } else {
        return {
            width: containerSize.width,
            height: containerSize.width / safeAspectRatio,
        };
    }
};
