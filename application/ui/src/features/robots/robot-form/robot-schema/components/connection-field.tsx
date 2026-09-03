import { ActionButton, ComboBox, Flex, Icon, Item, Text, View } from '@geti-ui/ui';
import { Refresh } from '@geti-ui/ui/icons';

import { getApiErrorMessage, isSerialPermissionDeniedError } from '../../../../../api/errors';
import { useCatalogIdentifyMutation, useDiscoverRobotsQuery } from '../../../robot-catalog.hooks';
import { SchemaRobotType } from '../../../robot-types';
import { InlineAlert } from '../../../setup-wizard/shared/inline-alert';
import { ConnectionItem } from '../types';

type Device = { serial_number: string | null; connection_string: string | null };

type ComboBoxFieldProps = {
    label: string;
    value: string;
    devices: Device[];
    allowsCustomValue: boolean;
    description?: string;
    onInputChange: (value: string) => void;
    onSelectionChange: (key: string | number | null) => void;
};

const normalizedSerialNumber = (serialNumber: string | null) =>
    serialNumber === 'no_serial' ? '' : (serialNumber ?? '');

const deviceKey = (device: Device) => {
    const serialNumber = normalizedSerialNumber(device.serial_number);
    return serialNumber ? `serial:${serialNumber}` : `port:${device.connection_string ?? ''}`;
};

const deviceTextValue = (device: Device) =>
    normalizedSerialNumber(device.serial_number) || device.connection_string || '';

const ComboBoxField = ({
    label,
    value,
    devices,
    allowsCustomValue,
    description,
    onInputChange,
    onSelectionChange,
}: ComboBoxFieldProps) => (
    <ComboBox
        label={label}
        description={description}
        width='100%'
        allowsCustomValue={allowsCustomValue}
        inputValue={value}
        onInputChange={onInputChange}
        onSelectionChange={onSelectionChange}
    >
        {devices.map((device) => (
            <Item key={deviceKey(device)} textValue={deviceTextValue(device)}>
                <Text>{device.serial_number ?? 'No serial number'}</Text>
                <Text slot='description'>{device.connection_string ?? ''}</Text>
            </Item>
        ))}
    </ComboBox>
);

type ConnectionFieldProps = {
    robotType: SchemaRobotType;
    payload: Record<string, unknown>;
    options: ConnectionItem;
    onChange: (field: string, value: unknown) => void;
};

const IdentifyError = ({ error }: { error: unknown }) => {
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

export const ConnectionField = ({ robotType, payload, options, onChange }: ConnectionFieldProps) => {
    const discover = useDiscoverRobotsQuery(robotType);
    const identify = useCatalogIdentifyMutation();
    const connectionKey = options.bind.connection;
    const serialNumberKey = options.bind.serial_number;
    const value = String(payload[connectionKey] ?? '');
    const setManualValue = (next: string) => {
        // Selecting a serial-capable device emits its serial number as an input change after selection.
        if ((discover.data ?? []).some((device) => device.serial_number !== null && deviceTextValue(device) === next)) {
            return;
        }
        onChange(connectionKey, next);
        if (serialNumberKey !== undefined) {
            onChange(serialNumberKey, '');
        }
    };

    return (
        <Flex direction='column' gap='size-100'>
            <Flex gap='size-100' alignItems='end'>
                <ComboBoxField
                    label={options.label ?? 'Connection'}
                    description={options.description}
                    value={value}
                    devices={discover.data ?? []}
                    allowsCustomValue={options.manual_entry !== false}
                    onInputChange={setManualValue}
                    onSelectionChange={(key) => {
                        const device = (discover.data ?? []).find((item) => deviceKey(item) === key);
                        if (device === undefined) {
                            return;
                        }
                        onChange(connectionKey, device.connection_string ?? '');
                        if (serialNumberKey !== undefined) {
                            onChange(serialNumberKey, normalizedSerialNumber(device.serial_number));
                        }
                    }}
                />
                <View>
                    <ActionButton onPress={() => discover.refetch()} isDisabled={discover.isFetching}>
                        <Icon>
                            <Refresh />
                        </Icon>
                    </ActionButton>
                </View>
                {options.identify && (
                    <View>
                        <ActionButton
                            onPress={() =>
                                identify.mutate({ params: { path: { robot_type: robotType } }, body: payload })
                            }
                            isDisabled={identify.isPending}
                        >
                            Identify
                        </ActionButton>
                    </View>
                )}
            </Flex>
            {identify.isError && <IdentifyError error={identify.error} />}
        </Flex>
    );
};
