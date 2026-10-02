import { InlineAlert } from '../shared/inline-alert';
import { ZeroCalibrationWizardContent } from '../zero-calibration/zero-calibration-wizard';

/**
 * reBot B601-RS setup: the shared zero-pose calibration wizard with RS guidance.
 * The plugin turns motor torque off before zeroing, so the arm goes limp.
 */
export const ReBotRSSetupWizardContent = () => (
    <ZeroCalibrationWizardContent
        title='Move the reBot B601-RS to its zero pose'
        tips={
            <InlineAlert variant='warning'>
                Motor torque is off during calibration, so the arm cannot hold itself up. Support it while you move it.
                Make sure the CAN interface is up at 1 Mbit/s before you begin.
            </InlineAlert>
        }
    />
);
