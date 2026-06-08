import { Divider, Flex, Heading, Item, Picker, Radio, RadioGroup, Text, View } from '@geti-ui/ui';
import { Label } from 'react-aria-components';

import { $api } from '../../api/client';
import { SchemaInferenceDeviceInfo, SchemaModel } from '../../api/openapi-spec';
import { INFERENCE_BACKENDS } from './inference-backends';

export const defaultBackend = 'openvino';

export type InferenceDevice = Pick<SchemaInferenceDeviceInfo, 'backend' | 'device'>;

const deviceTypeOrder: Record<SchemaInferenceDeviceInfo['type'], number> = {
    xpu: 0,
    npu: 1,
    cuda: 2,
    cpu: 3,
};

const getDeviceKey = ({ backend, device }: InferenceDevice) => `${backend}:${device}`;

const sortInferenceDevices = (a: SchemaInferenceDeviceInfo, b: SchemaInferenceDeviceInfo) => {
    const typeSort = deviceTypeOrder[a.type] - deviceTypeOrder[b.type];
    if (typeSort !== 0) {
        return typeSort;
    }

    return (a.index ?? Number.MAX_SAFE_INTEGER) - (b.index ?? Number.MAX_SAFE_INTEGER);
};

export const getSupportedInferenceDevices = (devices: SchemaInferenceDeviceInfo[], backend: string) => {
    return devices
        .filter((device) => {
            if (device.backend !== backend) {
                return false;
            }

            // We've not yet verified that running our models on NPU works,
            // so we filter it out for now
            if (device.type === 'npu') {
                return false;
            }

            // Using NVIDIA GPU in OpenVINO is not supported
            if (device.type === 'xpu' && device.name.includes('NVIDIA')) {
                return false;
            }

            return true;
        })
        .toSorted(sortInferenceDevices);
};

export const getDefaultInferenceDevice = (devices: SchemaInferenceDeviceInfo[], backend: string) => {
    const supportedDevices = getSupportedInferenceDevices(devices, backend);

    // Prefer hardware acceleration
    return supportedDevices.find((device) => device.type !== 'cpu') ?? supportedDevices.at(0);
};

interface BackendProps {
    id: string;
    isSelected?: boolean;
    isDisabled?: boolean;
}
const Backend = ({ id, isSelected = false, isDisabled = false }: BackendProps) => {
    const backend = INFERENCE_BACKENDS[id];

    return (
        <Label htmlFor={id}>
            <View
                borderColor={isSelected ? 'blue-400' : 'gray-50'}
                borderWidth='thick'
                borderRadius='small'
                padding='size-100'
                backgroundColor={isSelected ? 'gray-50' : 'gray-100'}
                UNSAFE_style={isDisabled ? { opacity: 0.7 } : undefined}
            >
                <Flex direction='row' gap='size-100' alignItems='center'>
                    <View padding='size-200'>
                        <Radio id={id} value={id} marginEnd={0} isDisabled={isDisabled} />
                    </View>

                    <Divider orientation='vertical' size='S' />
                    <Flex height='100%' alignItems='center' justifyContent={'center'} width='size-1000'>
                        <View padding='size-100'>
                            <backend.logo height={'50px'} />
                        </View>
                    </Flex>
                    <View>
                        <Heading level={4}>{backend.label}</Heading>
                        <Text
                            UNSAFE_style={{
                                fontSize: '12px',
                                color: 'var(--spectrum-global-color-gray-700)',
                            }}
                        >
                            {backend.description}
                        </Text>
                    </View>
                </Flex>
            </View>
        </Label>
    );
};

interface BackendSelectionProps {
    model: SchemaModel;
    inferenceDevice: InferenceDevice | undefined;
    inferenceDevices: SchemaInferenceDeviceInfo[];
    setInferenceDevice: (inferenceDevice: InferenceDevice | undefined) => void;
}

export const BackendSelection = ({
    model,
    inferenceDevice,
    inferenceDevices,
    setInferenceDevice,
}: BackendSelectionProps) => {
    const { data: policyBackends } = $api.useSuspenseQuery('get', '/api/policies/backends');
    const backend = inferenceDevice?.backend ?? defaultBackend;
    const device = inferenceDevice?.device;

    const backends: Array<string> =
        policyBackends?.[model.policy].filter((modelBackend) => ['openvino', 'torch'].includes(modelBackend)) ?? [];
    const devices = getSupportedInferenceDevices(inferenceDevices, backend);
    const selectedDevice =
        devices.find((availableDevice) => availableDevice.device === device) ??
        getDefaultInferenceDevice(inferenceDevices, backend);

    const onBackendChange = (value: string) => {
        const defaultDevice = getDefaultInferenceDevice(inferenceDevices, value);
        setInferenceDevice(defaultDevice);
    };

    return (
        <Flex gap='size-200' direction='column'>
            <RadioGroup value={backend} onChange={onBackendChange} isEmphasized width='100%'>
                <Flex gap='size-200' direction='column'>
                    {backends.map((backendId) => {
                        const isSelected = backendId === backend;
                        const isDisabled = !model.available_backends.includes(backendId);

                        return (
                            <Backend key={backendId} id={backendId} isSelected={isSelected} isDisabled={isDisabled} />
                        );
                    })}
                </Flex>
            </RadioGroup>
            <Picker
                width='100%'
                label='Inference device'
                selectedKey={selectedDevice === undefined ? undefined : getDeviceKey(selectedDevice)}
                onSelectionChange={(key) => {
                    const selected = devices.find((availableDevice) => getDeviceKey(availableDevice) === key);
                    if (selected !== undefined) {
                        setInferenceDevice(selected);
                    }
                }}
                isDisabled={devices.length === 0}
            >
                {devices.map((availableDevice) => (
                    <Item key={getDeviceKey(availableDevice)} textValue={availableDevice.name}>
                        <Text>{availableDevice.name}</Text>
                    </Item>
                ))}
            </Picker>
        </Flex>
    );
};
