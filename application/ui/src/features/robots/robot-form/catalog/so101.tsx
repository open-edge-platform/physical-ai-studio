import { Flex, Item, Picker, Text } from '@geti-ui/ui';
import { v4 as uuidv4 } from 'uuid';

import { $api } from '../../../../api/client';
import type { SchemaSo101RobotPayload } from '../../../../api/openapi-spec';
import type { SchemaRobot, SchemaRobotInput, SchemaRobotType } from '../../robot-types';
import { PermissionDeniedError } from '../../setup-wizard/so101/diagnostics-step-error';
import { useRobotFormFields } from '../provider';
import { IdentifyRobot, RefreshRobotsButton, useIdentifyMutation } from './actions';

export interface SO101FormData {
    name: string;
    payload: SchemaSo101RobotPayload;
}

export const getInitialSO101FormData = (robot?: SchemaRobot): SO101FormData => ({
    name: robot?.name ?? '',
    payload:
        robot && 'connection_string' in robot.payload ? robot.payload : { connection_string: '', serial_number: '' },
});

export const buildSO101Body = (
    formData: SO101FormData,
    schemaType: SchemaRobotType,
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
    const serialDevicesQuery = $api.useSuspenseQuery('get', '/api/hardware/serial_devices');

    const { formData, updateField, activeType } = useRobotFormFields<SO101FormData>();

    const identifyMutation = useIdentifyMutation();
    const identifyRobot = buildSO101Body(formData, activeType, uuidv4());
    const selectedKey =
        formData.payload.serial_number !== '' || formData.payload.connection_string !== ''
            ? getDeviceKey({
                  serial_number: formData.payload.serial_number,
                  connection_string: formData.payload.connection_string,
              })
            : null;

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
                        const device = serialDevicesQuery.data.find(
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
                    {serialDevicesQuery.data.map((serial_device) => {
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
                    <IdentifyRobot identifyMutation={identifyMutation} robot={identifyRobot} />
                </Flex>
            </Flex>

            {identifyMutation.isError && <PermissionDeniedError port={formData.payload.connection_string} />}
        </>
    );
};
