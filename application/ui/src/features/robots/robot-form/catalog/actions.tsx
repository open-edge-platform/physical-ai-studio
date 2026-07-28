import { ActionButton, Icon } from '@geti-ui/ui';
import { Refresh } from '@geti-ui/ui/icons';

import { useCatalogIdentifyMutation, useDiscoverRobotsQuery } from '../../robot-catalog.hooks';
import type { SchemaRobot, SchemaRobotType } from '../../robot-types';
import { useRobotFormFields } from '../provider';

import classes from '../form.module.css';

export const RefreshRobotsButton = () => {
    const { activeType } = useRobotFormFields();
    const { refetch, isFetching } = useDiscoverRobotsQuery(activeType);

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

export const IdentifyRobot = ({
    identifyMutation,
    payload: override,
    robotType,
}: {
    identifyMutation: ReturnType<typeof useCatalogIdentifyMutation>;
    payload?: SchemaRobot['payload'] | null;
    robotType?: SchemaRobotType;
}) => {
    const { formData, activeType } = useRobotFormFields();
    const payload = override ?? formData.payload;
    const isDisabled = payload === null || identifyMutation.isPending;

    const onIdentify = () => {
        if (isDisabled || payload === null) {
            return;
        }

        identifyMutation.mutate({
            params: { path: { robot_type: robotType ?? activeType } },
            body: payload,
        });
    };

    return (
        <ActionButton isDisabled={isDisabled} UNSAFE_className={classes.actionButton} onPress={onIdentify}>
            Identify
        </ActionButton>
    );
};
