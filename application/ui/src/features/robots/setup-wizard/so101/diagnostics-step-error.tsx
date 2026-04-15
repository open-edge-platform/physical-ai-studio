import { Button, Flex, Link, View } from '@geti-ui/ui';
import { useNavigate } from 'react-router';

import { InlineAlert } from '../shared/inline-alert';

import classes from '../shared/setup-wizard.module.scss';

const LEROBOT_DOCS_URL = 'https://huggingface.co/docs/lerobot/so101';

const PermissionDeniedError = ({ port }: { port: string | null }) => {
    const portDisplay = port ?? '/dev/ttyACM*';

    return (
        <>
            <InlineAlert variant='error'>
                <strong>Permission Denied</strong>
                <br />
                The application does not have permission to access the robot&apos;s USB port.
            </InlineAlert>
            <InlineAlert variant='info'>
                <strong>How to fix:</strong>
                <br />
                <strong>Option 1</strong>: Grant access to the port (temporary, resets on unplug):
                <View
                    backgroundColor={'gray-200'}
                    marginY='size-100'
                    paddingY='size-100'
                    paddingX='size-100'
                    borderRadius={'small'}
                >
                    <pre className={classes.codeBlock}>sudo chmod 666 {portDisplay}</pre>
                </View>
                <strong>Option 2</strong>: Add your user to the <code>dialout</code> group (permanent, requires logout):
                <View
                    backgroundColor={'gray-200'}
                    marginY='size-100'
                    paddingY='size-100'
                    paddingX='size-100'
                    borderRadius={'small'}
                >
                    <pre className={classes.codeBlock}>sudo usermod -aG dialout $USER</pre>
                </View>
                See the{' '}
                <Link href={LEROBOT_DOCS_URL} target='_blank' rel='noopener noreferrer'>
                    LeRobot SO101 docs
                </Link>{' '}
                for more details.
            </InlineAlert>
        </>
    );
};

const DeviceNotFoundError = ({ error }: { error: string }) => (
    <>
        <InlineAlert variant='error'>
            <strong>Device Not Found:</strong> {error}
        </InlineAlert>
        <InlineAlert variant='info'>
            <strong>Troubleshooting steps:</strong>
            <br />
            1. Check that the USB cable is firmly connected to both the robot controller and your computer.
            <br />
            2. Try a different USB port or cable.
            <br />
            3. Verify the robot&apos;s controller board has power (LED should be on).
            <br />
            4. Go back to robot discovery and re-scan for connected devices.
        </InlineAlert>
    </>
);

const ConnectionClosedError = () => (
    <InlineAlert variant='error'>
        <strong>Connection Lost:</strong> The connection to the robot was closed unexpectedly. This may indicate the USB
        cable was disconnected or the robot lost power. Please go back and try again.
    </InlineAlert>
);

const DefaultConnectionError = ({ error }: { error: string }) => (
    <InlineAlert variant='error'>
        <strong>Connection Error:</strong> {error}
    </InlineAlert>
);

interface DiagnosticsErrorProps {
    error: string;
    errorCode: string | null;
    port: string | null;
}

export const DiagnosticsError = ({ error, errorCode, port }: DiagnosticsErrorProps) => {
    const navigate = useNavigate();

    const errorContent = (() => {
        switch (errorCode) {
            case 'permission_denied':
                return <PermissionDeniedError port={port} />;
            case 'device_not_found':
                return <DeviceNotFoundError error={error} />;
            case 'connection_closed':
                return <ConnectionClosedError />;
            default:
                return <DefaultConnectionError error={error} />;
        }
    })();

    return (
        <Flex direction='column' gap='size-200'>
            {errorContent}

            <Flex gap='size-200'>
                <Button variant='secondary' onPress={() => navigate(-1)}>
                    Back
                </Button>
            </Flex>
        </Flex>
    );
};
