import { View } from '@geti-ui/ui';

import { ZeroCalibrationWizardContent } from '../../features/robots/setup-wizard/zero-calibration/zero-calibration-wizard';

/**
 * Route: /projects/:project_id/robots/new/zero-calibration
 *
 * Zero-pose calibration wizard for plugin robots whose catalog entry offers
 * calibration. Rendered as a child of NewRobotLayout, like the SO101 setup
 * wizard, so the robot form state is shared with the generic form.
 */
export const RobotZeroCalibration = () => {
    return (
        <View height='100%' backgroundColor='gray-100' padding='size-400' UNSAFE_style={{ overflow: 'hidden' }}>
            <ZeroCalibrationWizardContent />
        </View>
    );
};
