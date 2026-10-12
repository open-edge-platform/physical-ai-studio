import { View } from '@geti-ui/ui';

import { ReBotRSSetupWizardContent } from '../../features/robots/setup-wizard/rebot-rs/setup-wizard';

/**
 * Route: /projects/:project_id/robots/new/rebot-rs-setup
 *
 * Dedicated route for the reBot B601-RS zero-pose calibration wizard. Rendered as a
 * child of NewRobotLayout, like the SO101 setup wizard, so the robot form state
 * is shared with the generic form.
 */
export const ReBotRSSetup = () => {
    return (
        <View height='100%' backgroundColor='gray-100' padding='size-400' UNSAFE_style={{ overflow: 'hidden' }}>
            <ReBotRSSetupWizardContent />
        </View>
    );
};
