import { useEffect, useMemo, useState } from 'react';

import { DatasetCamera, useDatasetCameras } from '../../datasets/use-dataset-cameras';
import { getPolicyCameraSlots, PolicyCameraSlot } from './policy-camera-slots';

export interface FeatureMapping {
    /** The policy's camera slots, in slot order. Empty when the policy has no fixed order. */
    slots: PolicyCameraSlot[];
    /** Cameras the dataset was recorded with. */
    cameras: DatasetCamera[];
    /** Camera key filling each slot, keyed by slot id. */
    cameraKeyBySlotId: Record<string, string | null>;
    /** Put a camera in a slot, or clear the slot with `null`. */
    assignCamera: (slotId: string, cameraKey: string | null) => void;
    /** Dataset cameras that no slot uses — the policy would never see these. */
    unmappedCameras: DatasetCamera[];
    isLoading: boolean;
    /** The dataset has no episodes, so its cameras can't be read yet. */
    isDatasetEmpty: boolean;
    /** Why the mapping can't be trained with, or null when it is fine. */
    error: string | null;
    /**
     * Mapping in the shape the policy config takes: camera *name* -> slot index.
     *
     * Not the dataset's `observation.images.<name>` feature key: the policy sees its
     * images as `images.<name>` and prefixes `images.` onto whatever it is given, so a
     * full feature key would resolve to `images.observation.images.<name>` and match
     * nothing in the batch.
     */
    imageKeyReorderMap: Record<string, number>;
    /** Camera slots the policy reads, including the ones left empty. */
    numCameras: number;
}

/**
 * Maps the dataset's cameras onto the camera slots of the selected policy.
 *
 * Every dataset camera has to end up in a slot: the training side matches the
 * mapping against the batch's image keys and rejects a mapping that doesn't
 * cover them exactly. Slots may stay empty — those are filled with masked empty
 * images at training time.
 */
export const useFeatureMapping = (policy: string, datasetId: string | undefined): FeatureMapping => {
    const slots = useMemo(() => getPolicyCameraSlots(policy), [policy]);
    // A policy with no slots has nothing to map, so its dataset is never read.
    const { cameras, isLoading, isEmpty: isDatasetEmpty } = useDatasetCameras(slots.length > 0 ? datasetId : undefined);

    const [cameraKeyBySlotId, setCameraKeyBySlotId] = useState<Record<string, string | null>>({});

    // Default to the dataset's own order — camera 1 in the first slot, and so on —
    // and start over whenever the policy or the dataset's cameras change, since a
    // mapping only means something for the pair it was made for. The keys are joined
    // into one string so the effect has a stable dependency; a newline separates them
    // because a camera name may contain a space but never a line break.
    const cameraKeys = cameras.map((camera) => camera.key).join('\n');
    useEffect(() => {
        const keys = cameraKeys === '' ? [] : cameraKeys.split('\n');

        setCameraKeyBySlotId(Object.fromEntries(slots.map((slot, index) => [slot.id, keys.at(index) ?? null])));
    }, [slots, cameraKeys]);

    const assignCamera = (slotId: string, cameraKey: string | null) => {
        setCameraKeyBySlotId((current) => {
            const next = { ...current, [slotId]: cameraKey };

            // A camera belongs to one slot only, so moving it into a slot takes it
            // out of the slot it came from rather than duplicating it.
            if (cameraKey !== null) {
                for (const [otherSlotId, otherCameraKey] of Object.entries(current)) {
                    if (otherSlotId !== slotId && otherCameraKey === cameraKey) {
                        next[otherSlotId] = current[slotId];
                    }
                }
            }

            return next;
        });
    };

    const usedCameraKeys = new Set(Object.values(cameraKeyBySlotId).filter((key) => key !== null));
    const unmappedCameras = cameras.filter((camera) => !usedCameraKeys.has(camera.key));

    const missingRequiredSlot = slots.find((slot) => slot.isRequired && (cameraKeyBySlotId[slot.id] ?? null) === null);

    const error = (() => {
        if (slots.length === 0 || isLoading || isDatasetEmpty) {
            return null;
        }
        if (cameras.length > slots.length) {
            return (
                `This dataset has ${cameras.length} cameras but the policy reads only ${slots.length}. ` +
                'Train on a dataset with fewer cameras, or pick a policy that takes more.'
            );
        }
        if (unmappedCameras.length > 0) {
            const names = unmappedCameras.map((camera) => camera.name).join(', ');
            return `Every camera has to fill a slot; ${names} is not mapped yet.`;
        }
        if (missingRequiredSlot !== undefined) {
            return `The ${missingRequiredSlot.name} slot needs a camera.`;
        }
        return null;
    })();

    const imageKeyReorderMap = useMemo(() => {
        const map: Record<string, number> = {};

        slots.forEach((slot, index) => {
            const cameraKey = cameraKeyBySlotId[slot.id];
            const camera = cameras.find((candidate) => candidate.key === cameraKey);
            if (camera !== undefined) {
                map[camera.name] = index;
            }
        });

        return map;
    }, [slots, cameraKeyBySlotId, cameras]);

    return {
        slots,
        cameras,
        cameraKeyBySlotId,
        assignCamera,
        unmappedCameras,
        isLoading,
        isDatasetEmpty,
        error,
        imageKeyReorderMap,
        numCameras: slots.length,
    };
};
