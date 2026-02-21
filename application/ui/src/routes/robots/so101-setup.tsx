import { View } from '@geti/ui';
import { useSearchParams } from 'react-router-dom';

import { SchemaRobotType } from '../../api/openapi-spec';
import { RobotFormProvider } from '../../features/robots/robot-form/provider';
import { RobotModelsProvider } from '../../features/robots/robot-models-context';
import { SetupWizardContent } from '../../features/robots/so101/setup-wizard/setup-wizard';
import { SetupWizardProvider } from '../../features/robots/so101/setup-wizard/wizard-provider';

/**
 * Route: /projects/:project_id/robots/new/so101-setup
 *
 * Dedicated route for the SO101 multi-step setup wizard. Expects the robot's
 * initial form values (name, type, serial_number) as URL search params so
 * that state survives page refresh.
 *
 * Navigation flow:
 *   /robots/new  (generic form)  -->  /robots/new/so101-setup?name=...&type=...&serial_number=...
 */
export const SO101Setup = () => {
    const [searchParams] = useSearchParams();

    const initialRobot = {
        id: '',
        name: searchParams.get('name') ?? '',
        type: (searchParams.get('type') ?? 'SO101_Follower') as SchemaRobotType,
        serial_number: searchParams.get('serial_number') ?? '',
        connection_string: searchParams.get('connection_string') ?? '',
        active_calibration_id: null,
    };

    return (
        <RobotModelsProvider>
            <RobotFormProvider robot={initialRobot}>
                <SetupWizardProvider>
                    <View
                        height='100%'
                        backgroundColor='gray-100'
                        padding='size-400'
                        UNSAFE_style={{ overflow: 'hidden' }}
                    >
                        <SetupWizardContent />
                    </View>
                </SetupWizardProvider>
            </RobotFormProvider>
        </RobotModelsProvider>
    );
};
