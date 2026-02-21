import { Outlet } from 'react-router';

import { RobotFormProvider } from '../../features/robots/robot-form/provider';
import { RobotModelsProvider } from '../../features/robots/robot-models-context';

/**
 * Shared layout for the "Add new robot" flow.
 *
 * Wraps child routes with RobotModelsProvider and RobotFormProvider so that
 * form state (name, type, serial_number) is preserved when navigating between
 * the generic form (/robots/new) and the SO101 setup wizard
 * (/robots/new/so101-setup).
 */
export const NewRobotLayout = () => {
    return (
        <RobotModelsProvider>
            <RobotFormProvider>
                <Outlet />
            </RobotFormProvider>
        </RobotModelsProvider>
    );
};
