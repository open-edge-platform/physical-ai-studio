import { rs } from '@rstest/core';
import { screen } from '@testing-library/react';
import { HttpResponse } from 'msw';

import { SchemaSessionState } from '../../../api/openapi-spec';
import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { getMockedUserSettings } from '../../../test-utils/mocks/mock-settings';
import { render } from '../../../test-utils/render';
import { Statuses } from '../runtime-session-provider';
import { RecordMode, whyNotRecordable } from './record-mode';

const session: { state: SchemaSessionState; statuses: Statuses } = { state: {} as SchemaSessionState, statuses: {} };

rs.mock('../runtime-session-provider', () => ({
    ERROR_TOAST: {},
    failureOf: (statuses: Statuses, component: string) =>
        statuses[component]?.state === 'failed' ? (statuses[component].error ?? 'Failed') : null,
    useRuntimeSession: () => ({ ...session, send: rs.fn(async () => ({ ok: true })) }),
}));

const environment = {
    name: 'Bench',
    robot_id: 'r',
    leader_id: null,
    joint_names: [],
    cameras: [],
    environment_id: null,
};
const dataset = {
    dataset_id: 'd',
    name: 'Blocks',
    fps: 30,
    fps_fixed: false,
    is_recording: false,
    episodes_recorded: 0,
};
const teleop = { type: 'teleop' as const, hz: 30, grace_s: 3 };

const state = (overrides: Partial<SchemaSessionState>): SchemaSessionState => ({
    session_id: 's',
    environment,
    dataset,
    control: teleop,
    ...overrides,
});

const failed: Statuses = {
    environment: { type: 'status', component: 'environment', state: 'failed', error: 'cam 0: unplugged' },
};

describe('whyNotRecordable', () => {
    it('lets an episode start on a working environment, with or without a control', () => {
        expect(whyNotRecordable(state({}), {})).toBeNull();
        expect(whyNotRecordable(state({ control: null }), {})).toBeNull(); // stationary capture
        expect(whyNotRecordable(state({ dataset: null }), {})).toMatch(/Load a dataset/);
        expect(whyNotRecordable(state({ environment: null }), {})).toMatch(/Load an environment/);
        expect(whyNotRecordable(state({}), failed)).toBe(
            'The environment failed: cam 0: unplugged. Reload it to record.'
        );
    });
});

describe('RecordMode', () => {
    beforeEach(() => {
        server.use(
            http.get('/api/settings', () => HttpResponse.json(getMockedUserSettings())),
            http.get('/api/dataset/{dataset_id}', () => HttpResponse.json({ default_task: 'pick' } as never))
        );
    });

    it('disables Start and says why while the environment has failed', async () => {
        session.state = state({});
        session.statuses = failed;
        render(<RecordMode />);

        const start = await screen.findByRole('button', { name: /Start episode/ });
        expect(start).toBeDisabled();
        expect(screen.getByText(/The environment failed: cam 0: unplugged/)).toBeInTheDocument();
    });

    it('enables Start without a control, for stationary capture', async () => {
        session.state = state({ control: null });
        session.statuses = {};
        render(<RecordMode />);

        expect(await screen.findByRole('button', { name: /Start episode/ })).toBeEnabled();
    });
});
