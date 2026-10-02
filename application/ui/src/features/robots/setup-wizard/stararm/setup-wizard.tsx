import { InlineAlert } from '../shared/inline-alert';
import { ZeroCalibrationWizardContent } from '../zero-calibration/zero-calibration-wizard';

/**
 * Star Arm 102 leader setup: the shared zero-pose calibration wizard with leader guidance.
 * Teleoperation maps leader angles onto the follower, so both arms must share a zero pose.
 */
export const StarArmSetupWizardContent = () => (
    <ZeroCalibrationWizardContent
        title='Move the Star Arm leader to its zero pose'
        tips={
            <InlineAlert variant='info'>
                Use the same rest pose you calibrated the follower in, so the follower copies the leader exactly during
                teleoperation.
            </InlineAlert>
        }
    />
);
