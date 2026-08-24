import { Flex, Item, Picker, Text } from '@geti-ui/ui';

import { getApiErrorMessage, isSerialPermissionDeniedError } from '../../../../api/errors';
import type { SchemaSo101RobotPayload } from '../../../../api/openapi-spec';
import { useCatalogIdentifyMutation, useDiscoverRobotsQuery } from '../../robot-catalog.hooks';
import type { AvailableSchemaRobot, ConfigurableRobotType, SchemaRobotInput } from '../../robot-types';
import { InlineAlert } from '../../setup-wizard/shared/inline-alert';
import { PermissionDeniedError } from '../../setup-wizard/so101/diagnostics-step-error';
import { useRobotFormFields } from '../provider';
import { IdentifyRobot, RefreshRobotsButton } from './actions';

export interface SO101FormData {
    name: string;
    payload: SchemaSo101RobotPayload;
}

export const getInitialSO101FormData = (robot?: AvailableSchemaRobot): SO101FormData => ({
    name: robot?.name ?? '',
    payload:
        robot && (robot.type === 'SO101_Follower' || robot.type === 'SO101_Leader')
            ? robot.payload
            : { connection_string: '', serial_number: '' },
});

export const buildSO101Body = (
    formData: SO101FormData,
    schemaType: ConfigurableRobotType,
    robot_id: string
): SchemaRobotInput | null => {
    if (!formData.payload.serial_number && !formData.payload.connection_string) {
        return null;
    }

    return {
        id: robot_id,
        name: formData.name,
        type: schemaType,
        payload: formData.payload,
    } as SchemaRobotInput;
};

const getDeviceKey = ({
    serial_number,
    connection_string,
}: {
    serial_number: string;
    connection_string: string | null;
}) => {
    if (serial_number !== '') {
        return `serial:${serial_number}`;
    }
    return `port:${connection_string ?? ''}`;
};

export const SO101FormFields = () => {
    const { formData, updateField } = useRobotFormFields<SO101FormData>();

    const identifyMutation = useCatalogIdentifyMutation();
    const selectedKey =
        formData.payload.serial_number !== '' || formData.payload.connection_string !== ''
            ? getDeviceKey({
                  serial_number: formData.payload.serial_number,
                  connection_string: formData.payload.connection_string,
              })
            : null;

    const serialDevicesQuery = useDiscoverRobotsQuery('SO101_Follower');
    const devices = serialDevicesQuery.data ?? [];

    return (
        <>
            <Flex gap='size-100' justifyContent={'space-between'} alignItems={'end'}>
                <Picker
                    name='payload.device_key'
                    label='Select robot'
                    isRequired
                    width='100%'
                    selectedKey={selectedKey}
                    onSelectionChange={(key) => {
                        const device = devices.find(
                            (d) =>
                                getDeviceKey({
                                    serial_number: d.serial_number ?? '',
                                    connection_string: d.connection_string,
                                }) === String(key)
                        );

                        if (device === undefined) {
                            return;
                        }

                        const serial_number = device.serial_number ?? '';

                        updateField('payload', {
                            ...formData.payload,
                            serial_number,
                            connection_string: device.connection_string ?? '',
                        });
                    }}
                >
                    {devices.map((serial_device) => {
                        const serial_number = serial_device.serial_number ?? '';
                        const hasSerial = serial_number !== '';
                        const label = hasSerial ? serial_number : 'No serial number';

                        return (
                            <Item
                                key={getDeviceKey({
                                    serial_number,
                                    connection_string: serial_device.connection_string,
                                })}
                                textValue={label}
                            >
                                <Text>{label}</Text>
                                <Text slot='description'>{serial_device.connection_string ?? ''}</Text>
                            </Item>
                        );
                    })}
                </Picker>

                <Flex gap='size-100'>
                    <RefreshRobotsButton />
                    <IdentifyRobot identifyMutation={identifyMutation} />
                </Flex>
            </Flex>

            {identifyMutation.isError && (
                <IdentifyError error={identifyMutation.error} port={formData.payload.connection_string} />
            )}
        </>
    );
};

const IdentifyError = ({ error, port }: { error: unknown; port: string | null }) => {
    if (isSerialPermissionDeniedError(error)) {
        return <PermissionDeniedError port={port} />;
    }

    return (
        <InlineAlert variant='error'>
            <strong>Identify Failed</strong>
            <br />
            {getApiErrorMessage(error) ??
                'The robot could not be identified. Make sure it is powered on and not already in use, then try again.'}
        </InlineAlert>
    );
};
