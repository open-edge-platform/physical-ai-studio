import { ActionButton, Icon } from '@geti-ui/ui';
import { Refresh } from '@geti-ui/ui/icons';

import { $api } from '../../../../api/client';
import type { SchemaRobotInput } from '../../robot-types';

import classes from '../form.module.css';

export const RefreshRobotsButton = () => {
    const { refetch, isFetching } = $api.useSuspenseQuery('get', '/api/hardware/serial_devices');

    return (
        <ActionButton
            isDisabled={isFetching}
            UNSAFE_className={classes.actionButton}
            onPress={() => {
                refetch();
            }}
        >
            <Icon>
                <Refresh />
            </Icon>
        </ActionButton>
    );
};

export const useIdentifyMutation = () => {
    return $api.useMutation('post', '/api/hardware/identify', {
        meta: { skipInvalidation: true },
    });
};

export const IdentifyRobot = ({
    identifyMutation,
    robot,
}: {
    identifyMutation: ReturnType<typeof useIdentifyMutation>;
    robot: SchemaRobotInput | null;
}) => {
    const isDisabled = robot === null || identifyMutation.isPending;

    const onIdentify = () => {
        if (isDisabled || robot === null) {
            return;
        }

        identifyMutation.mutate({ body: robot });
    };

    return (
        <ActionButton isDisabled={isDisabled} UNSAFE_className={classes.actionButton} onPress={onIdentify}>
            Identify
        </ActionButton>
    );
};
