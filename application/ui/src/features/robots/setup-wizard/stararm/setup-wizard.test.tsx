import { screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { render } from '../../../../test-utils/render';
import { RobotFormProvider } from '../../robot-form/provider';
import { useZeroCalibrationWebSocket } from '../zero-calibration/use-zero-calibration-websocket';
import { StarArmSetupWizardContent } from './setup-wizard';

vi.mock('../zero-calibration/use-zero-calibration-websocket', () => ({
    useZeroCalibrationWebSocket: vi.fn(),
}));

// The 3D viewer needs WebGL.
vi.mock('../shared/setup-robot-viewer', () => ({
    SetupRobotViewer: () => <div>robot-viewer</div>,
}));
vi.mock('../../use-joint-state', () => ({
    useSynchronizeModelJoints: vi.fn(),
}));

describe('StarArmSetupWizardContent', () => {
    it('shows its own title and guidance over the shared calibration wizard', async () => {
        vi.mocked(useZeroCalibrationWebSocket).mockReturnValue({
            state: {
                phase: 'positioning',
                statusMessage: 'Plugin instructions.',
                joints: null,
                calibrationResult: null,
                isSettingZero: false,
                error: null,
                errorCode: null,
                isConnected: true,
            },
            readyState: 1,
            commands: { setZero: vi.fn() },
        });

        render(
            <RobotFormProvider robot={{ type: 'StarArm_102_HD_Leader', name: 'Khaos', payload: {} }}>
                <StarArmSetupWizardContent />
            </RobotFormProvider>,
            { route: '/projects/project-1/robots/new', path: '/projects/:project_id/robots/new' }
        );

        expect(
            await screen.findByRole('heading', { name: 'Move the Star Arm leader to its zero pose' })
        ).toBeInTheDocument();
        expect(screen.getByText(/same rest pose you calibrated the follower in/)).toBeInTheDocument();
        expect(screen.getByText('Plugin instructions.')).toBeInTheDocument();
    });
});
