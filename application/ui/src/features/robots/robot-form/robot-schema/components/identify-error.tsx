import { getApiErrorMessage, isSerialPermissionDeniedError } from '../../../../../api/errors';
import { InlineAlert } from '../../../setup-wizard/shared/inline-alert';

export const IdentifyError = ({ error }: { error: unknown }) => {
    if (isSerialPermissionDeniedError(error)) {
        return (
            <InlineAlert variant='error'>
                <strong>Permission Denied</strong>: The application does not have permission to access the robot&apos;s
                USB port.
            </InlineAlert>
        );
    }

    return (
        <InlineAlert variant='error'>
            <strong>Identify Failed</strong>:{' '}
            {getApiErrorMessage(error) ??
                'The robot could not be identified. Make sure it is powered on and not already in use, then try again.'}
        </InlineAlert>
    );
};
