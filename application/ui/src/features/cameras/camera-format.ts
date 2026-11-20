// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { SchemaProjectCamera } from '../../api/types';

/** A camera that Studio streams from itself, as opposed to a simulation camera, which its simulation streams. */
export type StreamedCamera = Exclude<SchemaProjectCamera, { driver: 'simulation' }>;

export const isSimulationCamera = (
    camera: SchemaProjectCamera
): camera is Extract<SchemaProjectCamera, { driver: 'simulation' }> => camera.driver === 'simulation';

/** "640 x 480 @ 30", or null for a simulation camera: its simulation decides while it runs. */
export const formatCameraMode = (camera: SchemaProjectCamera): string | null =>
    isSimulationCamera(camera) ? null : `${camera.payload.width} x ${camera.payload.height} @ ${camera.payload.fps}`;
