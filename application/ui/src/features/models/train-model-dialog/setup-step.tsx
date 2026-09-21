import { Flex, Item, Key, Picker } from '@geti-ui/ui';

import { SchemaDeviceInfo, SchemaProjectInput } from '../../../api/openapi-spec';
import { InlineAlert } from '../../robots/setup-wizard/shared/inline-alert';
import { PolicyAccessAlert } from './policy-access-alert';
import { PolicySelection } from './policy-selection';

interface TrainingTargetOption {
    id: string;
    label: string;
}

interface SetupStepProps {
    datasets: SchemaProjectInput['datasets'];
    selectedDataset: Key | null;
    onSelectedDatasetChange: (value: Key | null) => void;
    trainingTargetOptions: TrainingTargetOption[];
    remoteTrainerId: Key | null;
    onRemoteTrainerIdChange: (value: Key | null) => void;
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
    remoteTrainerId,
    onRemoteTrainerIdChange,
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

        <Picker
            label='Run on'
            selectedKey={remoteTrainerId}
            onSelectionChange={onRemoteTrainerIdChange}
            width='100%'
            items={trainingTargetOptions}
        >
            {(trainingTarget) => <Item key={trainingTarget.id}>{trainingTarget.label}</Item>}
        </Picker>

        <Picker label='Dataset' selectedKey={selectedDataset} onSelectionChange={onSelectedDatasetChange} width='100%'>
            {datasets.map((dataset) => (
                <Item key={dataset.id}>{dataset.name}</Item>
            ))}
        </Picker>

        <PolicySelection
            selectedPolicy={selectedPolicy}
            onSelectionChange={onSelectedPolicyChange}
            isDisabled={isPolicyDisabled}
            trainingDevice={activeDevice}
        />
        <PolicyAccessAlert policy={selectedPolicy} />
    </Flex>
);
