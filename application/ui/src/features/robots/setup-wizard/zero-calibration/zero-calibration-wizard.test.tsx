import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { http } from '../../../../api/utils';
import { server } from '../../../../msw-node-setup';
import { render } from '../../../../test-utils/render';
import { RobotFormProvider } from '../../robot-form/provider';
import { useZeroCalibrationWebSocket, ZeroCalibrationWebSocketState } from './use-zero-calibration-websocket';
import { ZeroCalibrationWizardContent } from './zero-calibration-wizard';

vi.mock('./use-zero-calibration-websocket', () => ({
    useZeroCalibrationWebSocket: vi.fn(),
}));

// The 3D viewer needs WebGL; the live sync is covered by use-joint-state.
vi.mock('../shared/setup-robot-viewer', () => ({
    SetupRobotViewer: () => <div>robot-viewer</div>,
}));
vi.mock('../../use-joint-state', () => ({
    useSynchronizeModelJoints: vi.fn(),
}));

const PROJECT_ID = 'project-1';
const setZero = vi.fn();

const READY_STATE: ZeroCalibrationWebSocketState = {
    phase: 'positioning',
    statusMessage: 'Fold the arm and close the gripper.',
    joints: { 'shoulder_pan.pos': 40 },
    calibrationResult: null,
    isSettingZero: false,
    error: null,
    errorCode: null,
    isConnected: true,
};

const renderWizard = (state: Partial<ZeroCalibrationWebSocketState> = {}) => {
    vi.mocked(useZeroCalibrationWebSocket).mockReturnValue({
        state: { ...READY_STATE, ...state },
        readyState: 1,
        commands: { setZero },
    });

    return render(
        <RobotFormProvider
            robot={{ type: 'ReBot_B601_RS_Follower', name: 'Khaos', payload: { connection_string: 'can0' } }}
        >
            <ZeroCalibrationWizardContent />
        </RobotFormProvider>,
        {
            route: `/projects/${PROJECT_ID}/robots/new/zero-calibration`,
            path: '/projects/:project_id/robots/new/zero-calibration',
        }
    );
};

describe('ZeroCalibrationWizardContent', () => {
    beforeEach(() => {
        setZero.mockClear();
    });

    it('starts calibration with the robot from the form', () => {
        renderWizard();

        expect(vi.mocked(useZeroCalibrationWebSocket)).toHaveBeenCalledWith({
            projectId: PROJECT_ID,
            robot: expect.objectContaining({
                name: 'Khaos',
                type: 'ReBot_B601_RS_Follower',
                payload: { connection_string: 'can0' },
            }),
        });
    });

    it('shows the plugin instructions and sets zero on request', async () => {
        const user = userEvent.setup();
        renderWizard();

        expect(await screen.findByText('Fold the arm and close the gripper.')).toBeInTheDocument();
        await user.click(screen.getByRole('button', { name: 'Set zero' }));

        expect(setZero).toHaveBeenCalledTimes(1);
    });

    it('disables set zero until the arm is connected and released', async () => {
        renderWizard({ phase: 'connecting' });

        expect(await screen.findByRole('button', { name: 'Set zero' })).toBeDisabled();
    });

    it('asks to retry when a joint does not read zero', async () => {
        renderWizard({
            calibrationResult: {
                event: 'calibration_result',
                success: false,
                joints: { 'shoulder_pan.pos': 0.4, 'gripper.pos': 12.3 },
                tolerance_deg: 5,
            },
        });

        expect(await screen.findByText(/Some joints do not read zero/)).toBeInTheDocument();
        expect(screen.getByText('12.3°')).toBeInTheDocument();
        expect(screen.getByText('Off zero')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Set zero' })).toBeInTheDocument();
    });

    it('saves the robot once calibration succeeds', async () => {
        const user = userEvent.setup();
        const created = vi.fn();
        server.use(
            http.post('/api/projects/{project_id}/robots', async ({ request }) => {
                const body = await request.json();
                created(body);
                return HttpResponse.json({ ...(body as object), connection_status: 'unknown' } as never);
            })
        );
        renderWizard({
            phase: 'verification',
            calibrationResult: {
                event: 'calibration_result',
                success: true,
                joints: { 'shoulder_pan.pos': 0.4 },
                tolerance_deg: 5,
            },
        });

        expect(await screen.findByText(/Zero position set/)).toBeInTheDocument();
        await user.click(screen.getByRole('button', { name: 'Save Robot' }));

        await vi.waitFor(() => expect(created).toHaveBeenCalledTimes(1));
        expect(created.mock.calls[0][0]).toMatchObject({ name: 'Khaos', type: 'ReBot_B601_RS_Follower' });
    });

    it('lets an already calibrated arm be added without a new zero', async () => {
        const user = userEvent.setup();
        const created = vi.fn();
        server.use(
            http.post('/api/projects/{project_id}/robots', async ({ request }) => {
                const body = await request.json();
                created(body);
                return HttpResponse.json({ ...(body as object), connection_status: 'unknown' } as never);
            })
        );
        renderWizard();

        await user.click(await screen.findByRole('button', { name: 'Skip calibration' }));

        await vi.waitFor(() => expect(created).toHaveBeenCalledTimes(1));
        expect(setZero).not.toHaveBeenCalled();
    });
});
