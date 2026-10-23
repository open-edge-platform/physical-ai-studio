// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

export const fps = 30;
export const sceneWidth = 1600;
export const sceneHeight = 900;

/** Return linear, clamped progress on the explicit frame timeline. */
export const progress = (frame: number, start: number, end: number): number =>
  Math.max(0, Math.min(1, (frame - start) / (end - start)));
