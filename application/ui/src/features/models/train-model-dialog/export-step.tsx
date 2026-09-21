import { Checkbox, CheckboxGroup, Flex, ProgressCircle, Text } from '@geti-ui/ui';

import { InlineAlert } from '../../robots/setup-wizard/shared/inline-alert';
import { isExportBackend } from '../inference-backends';
import { InferenceBackendLogo } from '../model-formats/backend-card';
import { MODELS } from './policies';
import { ExportSelection } from './use-export-backends';

import classes from './train-model-dialog.module.css';

interface ExportStepProps {
    policy: string;
    selection: ExportSelection;
}

/**
 * Picks the formats the trained model is exported to.
 *
 * Training exports the model once it is done, and each format is a separate
 * conversion of the same weights — so this is about which runtimes the model
 * should be deployable on, not about how it is trained.
 */
export const ExportStep = ({ policy, selection }: ExportStepProps) => {
    const policyName = MODELS.find((model) => model.id === policy)?.name ?? policy;
    const { backends, selectedBackends, setSelectedBackends, isLoading, error } = selection;

    if (isLoading) {
        return (
            <Flex alignItems='center' justifyContent='center' height='100%' width='100%'>
                <ProgressCircle aria-label='Loading export formats' isIndeterminate />
            </Flex>
        );
    }

    if (backends.length === 0) {
        return (
            <Flex direction='column' justifyContent='center' height='100%' width='100%'>
                <Text>{policyName} has no export formats, so the trained model stays a checkpoint.</Text>
            </Flex>
        );
    }

    return (
        <Flex direction='column' gap='size-200' width='100%'>
            <Text>
                Export the trained {policyName} model to the runtimes you want to deploy it on. Each format is converted
                after training.
            </Text>

            <CheckboxGroup
                isEmphasized
                width='100%'
                aria-label='Export formats'
                value={selectedBackends}
                onChange={(selected) => setSelectedBackends(selected.filter(isExportBackend))}
            >
                {backends.map((backend) => (
                    <Checkbox key={backend.type} value={backend.type} UNSAFE_className={classes.exportFormatOption}>
                        <InferenceBackendLogo backend={backend} isAvailable />
                    </Checkbox>
                ))}
            </CheckboxGroup>

            {error !== null && <InlineAlert variant='warning'>{error}</InlineAlert>}
        </Flex>
    );
};
