import { Flex, Item, Key, Picker, StatusLight, Text } from '@geti-ui/ui';

import { SchemaDeviceInfo, SchemaProjectInput } from '../../../api/openapi-spec';
import { InlineAlert } from '../../robots/setup-wizard/shared/inline-alert';
import { formatBytes } from './policies';
import { PolicyAccessAlert } from './policy-access-alert';
import { PolicySelection } from './policy-selection';
import { TrainingTargetOption } from './train-model-dialog';

interface SetupStepProps {
    datasets: SchemaProjectInput['datasets'];
    selectedDataset: Key | null;
    onSelectedDatasetChange: (value: Key | null) => void;
    trainingTargetOptions: TrainingTargetOption[];
    targetId: Key | null;
    onTargetIdChange: (value: Key | null) => void;
    remoteDevices: SchemaDeviceInfo[];
    busyGpuKeys: Set<string>;
    selectedGpuKey: Key | null;
    onSelectedGpuKeyChange: (value: Key | null) => void;
    selectedGpuUnavailable: boolean;
    remoteUnavailable: boolean;
    selectedPolicy: string;
    onSelectedPolicyChange: (policy: string) => void;
    isPolicyDisabled: boolean;
    activeDevice: SchemaDeviceInfo | null;
}

export const SetupStep = ({
    datasets,
    selectedDataset,
    onSelectedDatasetChange,
    trainingTargetOptions,
    targetId,
    onTargetIdChange,
    remoteDevices,
    busyGpuKeys,
    selectedGpuKey,
    onSelectedGpuKeyChange,
    selectedGpuUnavailable,
    remoteUnavailable,
    selectedPolicy,
    onSelectedPolicyChange,
    isPolicyDisabled,
    activeDevice,
}: SetupStepProps) => (
    <Flex direction='column' gap='size-200' width='100%'>
        {remoteUnavailable && (
            <InlineAlert variant='warning'>
                Can&apos;t reach the remote trainer, so training can&apos;t start. Make sure it&apos;s running, then try
                again.
            </InlineAlert>
        )}

        <Picker label='Dataset' selectedKey={selectedDataset} onSelectionChange={onSelectedDatasetChange} width='100%'>
            {datasets.map((dataset) => (
                <Item key={dataset.id}>{dataset.name}</Item>
            ))}
        </Picker>

        <Picker
            label='Run on'
            selectedKey={targetId}
            onSelectionChange={onTargetIdChange}
            width='100%'
            items={trainingTargetOptions}
        >
            {(trainingTarget) => (
                <Item key={trainingTarget.id} textValue={trainingTarget.label}>
                    <Text>{trainingTarget.label}</Text>
                    {/* `Item` only recognizes plain `Text` children for its label/description
                        slots (a `StatusLight` isn't one), so nest it inside the description
                        slot rather than passing it as a sibling — otherwise both children
                        collapse into the same "label" grid area and overlap. */}
                    <Text slot='description'>{trainingTarget.statusLabel}</Text>
                    <Text slot='icon'>
                        <StatusLight variant={trainingTarget.statusVariant} marginBottom={0} />
                    </Text>
                </Item>
            )}
        </Picker>

        {(remoteDevices.length > 1 || selectedGpuKey !== null) && (
            <Picker
                label='GPU'
                selectedKey={selectedGpuKey ?? (activeDevice ? `${activeDevice.type}:${activeDevice.index}` : null)}
                onSelectionChange={onSelectedGpuKeyChange}
                width='100%'
                items={remoteDevices.map((device) => ({
                    key: `${device.type}:${device.index}`,
                    label: `${device.type.toUpperCase()} ${device.index} — ${device.name}${
                        device.memory ? ` (${formatBytes(device.memory)})` : ''
                    }`,
                    busy: busyGpuKeys.has(`${device.type}:${device.index}`),
                }))}
            >
                {(device) => (
                    <Item key={device.key} textValue={device.label}>
                        <Text>{device.label}</Text>
                        <Text slot='icon'>
                            <StatusLight
                                variant={device.busy ? 'yellow' : 'positive'}
                                role='status'
                                aria-label={device.busy ? 'GPU busy; new jobs will wait' : 'GPU free'}
                                marginBottom={0}
                            />
                        </Text>
                    </Item>
                )}
            </Picker>
        )}
        {selectedGpuUnavailable && (
            <InlineAlert variant='warning'>The selected GPU is no longer available. Choose another GPU.</InlineAlert>
        )}

        <PolicySelection
            selectedPolicy={selectedPolicy}
            onSelectionChange={onSelectedPolicyChange}
            isDisabled={isPolicyDisabled}
            trainingDevice={activeDevice}
        />
        <PolicyAccessAlert policy={selectedPolicy} />
    </Flex>
);
