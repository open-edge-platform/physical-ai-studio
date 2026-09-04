import { ActionButton, Flex, TextField, View } from '@geti-ui/ui';

import { useCatalogIdentifyMutation } from '../../../robot-catalog.hooks';
import { SchemaRobotType } from '../../../robot-types';
import { IpAddressItem } from '../types';
import { IdentifyError } from './identify-error';

type IpAddressFieldProps = {
    robotType: SchemaRobotType;
    payload: Record<string, unknown>;
    options: IpAddressItem;
    isRequired: boolean;
    onChange: (field: string, value: unknown) => void;
};

export const IpAddressField = ({ robotType, payload, options, isRequired, onChange }: IpAddressFieldProps) => {
    const identify = useCatalogIdentifyMutation();
    const value = String(payload[options.name] ?? '');
    const identifyValue = value.trim();
    const identifyRobotType = options.identify_robot_type ?? robotType;
    const identifyPayload =
        options.identify_robot_type === undefined
            ? { ...payload, [options.name]: identifyValue }
            : { connection_string: identifyValue };

    return (
        <Flex direction='column' gap='size-100'>
            <Flex gap='size-100' alignItems='end'>
                <TextField
                    isRequired={isRequired}
                    label={options.label ?? 'IP address'}
                    description={options.description}
                    width='100%'
                    value={value}
                    onChange={(next) => onChange(options.name, next)}
                />
                {options.identify && (
                    <View>
                        <ActionButton
                            isDisabled={identifyValue === '' || identify.isPending}
                            onPress={() =>
                                identify.mutate({
                                    params: { path: { robot_type: identifyRobotType } },
                                    body: identifyPayload,
                                })
                            }
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
