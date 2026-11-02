import { Outlet } from 'react-router';

import { RobotFormProvider } from '../../features/robots/robot-form/provider';
import { RobotModelsProvider } from '../../features/robots/robot-models-context';

/**
 * Shared layout for the "Configure new robot" flow.
 *
 * Wraps child routes with RobotModelsProvider and RobotFormProvider so that
 * form state (name, type, serial_number) is preserved when navigating between
 * the generic form (/robots/new), the SO101 setup wizard
 * (/robots/new/so101-setup) and the zero-pose calibration wizards
 * (/robots/new/rebot-rs-setup, /robots/new/stararm-setup and, for other
 * plugins, /robots/new/zero-calibration).
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
