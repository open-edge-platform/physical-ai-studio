import { View } from '@geti-ui/ui';

import { StarArmSetupWizardContent } from '../../features/robots/setup-wizard/stararm/setup-wizard';

/**
 * Route: /projects/:project_id/robots/new/stararm-setup
 *
 * Dedicated route for the Star Arm 102 leader zero-pose calibration wizard. Rendered as a
 * child of NewRobotLayout, like the SO101 setup wizard, so the robot form state
 * is shared with the generic form.
 */
export const StarArmSetup = () => {
    return (
        <View height='100%' backgroundColor='gray-100' padding='size-400' UNSAFE_style={{ overflow: 'hidden' }}>
            <StarArmSetupWizardContent />
        </View>
    );
};
