// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { StatusLight } from '@geti-ui/ui';

import { SchemaAutopilotRecordingState } from '../../api/openapi-spec';
import { isAutopilotRecording } from './automatic-recording';

type Variant = 'positive' | 'negative' | 'notice' | 'info' | 'neutral';

/** What the simulation's autopilot does for the session, in a few words, from its autopilot recording. */
export const describeAutopilot = (
    recording: SchemaAutopilotRecordingState | null | undefined
): { variant: Variant; text: string } => {
    if (recording == null || !isAutopilotRecording(recording)) {
        return { variant: 'neutral', text: 'off, it leads while you record automatically' };
    }
    switch (recording.phase) {
        case 'preparing':
            return { variant: 'info', text: 'laying out the next episode' };
        case 'recording':
            return { variant: 'info', text: recording.task ? `leading: ${recording.task}` : 'leading' };
        case 'saving':
            return { variant: 'positive', text: 'episode done, saving it' };
        case 'stopping':
            return { variant: 'notice', text: 'stopping' };
        default:
            return { variant: 'info', text: 'starting' };
    }
};

/** The simulation's autopilot, as the session's autopilot recording reports it: no polling of the simulation. */
export const AutopilotStatus = ({ recording }: { recording: SchemaAutopilotRecordingState | null | undefined }) => {
    const { variant, text } = describeAutopilot(recording);

    return <StatusLight variant={variant}>Autopilot: {text}</StatusLight>;
};
