import { View } from '@geti-ui/ui';

import { useRobotForm } from '../../robot-form/provider';
import { InlineAlert } from '../shared/inline-alert';
import { ZeroCalibrationWizardContent } from '../zero-calibration/zero-calibration-wizard';

import classes from '../shared/setup-wizard.module.css';

const DEFAULT_CAN_INTERFACE = 'can0';

const canSetupCommands = (
    canInterface: string
) => `# The kit includes PCAN-USB, which should normally show up as can0 or can1
sudo modprobe peak_usb
ip -br link

# If ${canInterface} appears, set the bitrate
sudo ip link set ${canInterface} down 2>/dev/null
sudo ip link set ${canInterface} type can bitrate 1000000
sudo ip link set ${canInterface} up`;

const CanSetupTip = () => {
    const { payload } = useRobotForm();
    const connectionString = payload.connection_string;
    const canInterface =
        typeof connectionString === 'string' && connectionString.trim() !== ''
            ? connectionString.trim()
            : DEFAULT_CAN_INTERFACE;

    return (
        <InlineAlert variant='info'>
            <strong>Before you begin:</strong>
            <br />
            The CAN interface <code>{canInterface}</code> must be up at 1 Mbit/s before the arm can be discovered. Run
            this in a terminal:
            <View
                backgroundColor={'gray-200'}
                marginY='size-100'
                paddingY='size-100'
                paddingX='size-100'
                borderRadius={'small'}
            >
                <pre className={classes.codeBlock}>{canSetupCommands(canInterface)}</pre>
            </View>
        </InlineAlert>
    );
};

/**
 * reBot B601-RS setup: the shared zero-pose calibration wizard with RS guidance.
 * The plugin turns motor torque off before zeroing, so the arm goes limp.
 */
export const ReBotRSSetupWizardContent = () => (
    <ZeroCalibrationWizardContent
        title='Move the reBot B601-RS to its zero pose'
        tips={
            <>
                <CanSetupTip />
                <InlineAlert variant='warning'>
                    Motor torque is off during calibration, so the arm cannot hold itself up. Support it while you move
                    it.
                </InlineAlert>
            </>
        }
    />
);
