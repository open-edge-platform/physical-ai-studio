import { Button, Heading, IllustratedMessage, View } from '@geti-ui/ui';
import { AlertCircle as NotFound } from '@geti-ui/ui/icons';
import { isRouteErrorResponse, useNavigate, useRouteError } from 'react-router';

import { getApiErrorMessage } from '../../api/errors';
import { paths } from '../../router';
import { useProjectId } from '../projects/use-project';

const isNotFound = (error: unknown): boolean =>
    isRouteErrorResponse(error)
        ? error.status === 404
        : typeof error === 'object' &&
          error !== null &&
          'http_status' in error &&
          (error as Record<string, unknown>).http_status === 404;

export const RobotDetailError = () => {
    const error = useRouteError();
    const navigate = useNavigate();
    const { project_id } = useProjectId();

    const heading = isNotFound(error)
        ? 'This robot no longer exists.'
        : (getApiErrorMessage(error) ?? 'Something went wrong while loading this robot.');

    return (
        <View height='100%' flex='1'>
            <IllustratedMessage>
                <NotFound />
                <Heading>{heading}</Heading>
                <Button
                    variant='accent'
                    marginTop='size-200'
                    onPress={() => navigate(paths.project.robots.index({ project_id }))}
                >
                    Back to robots
                </Button>
            </IllustratedMessage>
        </View>
    );
};
