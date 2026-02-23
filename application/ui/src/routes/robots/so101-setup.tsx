import { View } from '@geti/ui';

import { SetupWizardContent } from '../../features/robots/setup-wizard/so101/setup-wizard';
import { SetupWizardProvider } from '../../features/robots/setup-wizard/so101/wizard-provider';

/**
 * Route: /projects/:project_id/robots/new/so101-setup
 *
 * Dedicated route for the SO101 multi-step setup wizard. Rendered as a child
 * of NewRobotLayout which provides RobotFormProvider and RobotModelsProvider,
 * so form state (name, type, serial_number) is shared with the generic form
 * and preserved across navigation.
 */
export const SO101Setup = () => {
    return (
        <SetupWizardProvider>
            <View height='100%' backgroundColor='gray-100' padding='size-400' UNSAFE_style={{ overflow: 'hidden' }}>
                <SetupWizardContent />
            </View>
        </SetupWizardProvider>
    );
};
