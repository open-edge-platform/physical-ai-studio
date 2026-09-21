import { Flex, Item, Key, Picker, StatusLight, Text } from '@geti-ui/ui';

import { SchemaDeviceInfo, SchemaProjectInput } from '../../../api/openapi-spec';
import { InlineAlert } from '../../robots/setup-wizard/shared/inline-alert';
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
    remoteUnavailable: boolean;
    sshUnavailable: boolean;
    sshUnverified: boolean;
    sshStatusMessage: string;
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
    remoteUnavailable,
    sshUnavailable,
    sshUnverified,
    sshStatusMessage,
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

        {sshUnavailable && (
            <InlineAlert variant='warning'>
                This remote server isn&apos;t ready for training (status: {sshStatusMessage}). Verify the server before
                submitting a job.
            </InlineAlert>
        )}

        {sshUnverified && (
            <InlineAlert variant='info'>
                This remote server hasn&apos;t been verified yet. Submitting will pull and verify the trainer image
                first, which can take a few minutes.
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
                    <Text slot={'icon'}>
                        <StatusLight variant={trainingTarget.statusVariant} marginBottom={0} />
                    </Text>
                </Item>
            )}
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
