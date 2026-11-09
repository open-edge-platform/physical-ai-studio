// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { SchemaUsbCameraOutput } from '../../api/openapi-spec';

export const getMockedProjectCamera = (overrides: Partial<SchemaUsbCameraOutput> = {}): SchemaUsbCameraOutput => ({
    id: 'camera-1',
    driver: 'usb_camera',
    name: 'Front camera',
    fingerprint: null,
    hardware_name: null,
    payload: { width: 640, height: 480, fps: 30 },
    ...overrides,
});
