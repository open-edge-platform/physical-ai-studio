import { useState } from 'react';

import { Button, ButtonGroup, Content, Dialog, Divider, Flex, Heading, RadioGroup, Text } from '@geti-ui/ui';

import { getApiErrorMessage } from '../../../api/errors';
import type { SchemaModelExportAction } from '../../../api/openapi-spec';
import { BackendOption } from '../backend-selection/backend-option';

interface AddModelFormatDialogProps {
    actions: SchemaModelExportAction[];
    initialAction?: SchemaModelExportAction;
    isSubmitting: boolean;
    isExportDisabled: boolean;
    error: unknown;
    onExport: (action: SchemaModelExportAction) => void;
    close: () => void;
}

export const AddModelFormatDialog = ({
    actions,
    initialAction,
    isSubmitting,
    isExportDisabled,
    error,
    onExport,
    close,
}: AddModelFormatDialogProps) => {
    const [selection, setSelection] = useState(initialAction ?? actions[0]);
    const [hasSubmitted, setHasSubmitted] = useState(false);
    const isCompression = selection?.operation === 'int8_sym';

    return (
        <Dialog>
            <Heading>Add model format</Heading>
            <Divider />
            <Content>
                <Flex direction='column' gap='size-200'>
                    <Text>
                        Export the stored checkpoint or create an optimized model format. Existing model formats stay
                        unchanged.
                    </Text>
                    <RadioGroup
                        label='Model format'
                        value={selection?.backend}
                        onChange={(value) => {
                            const action = actions.find((option) => option.backend === value);
                            if (action) {
                                setSelection(action);
                                setHasSubmitted(false);
                            }
                        }}
                        isDisabled={isExportDisabled}
                        isEmphasized
                        width='100%'
                    >
                        <Flex direction='column' gap='size-200'>
                            {actions.map((action) => (
                                <BackendOption
                                    key={`${action.backend}/${action.operation}`}
                                    id={action.backend}
                                    isSelected={action.backend === selection?.backend}
                                    label={action.operation === 'int8_sym' ? 'OpenVINO · INT8' : undefined}
                                    description={
                                        action.operation === 'int8_sym'
                                            ? 'Compress weights to INT8. The original export stays unchanged.'
                                            : undefined
                                    }
                                />
                            ))}
                        </Flex>
                    </RadioGroup>
                    <Text
                        UNSAFE_style={{
                            fontSize: '12px',
                            color: 'var(--spectrum-global-color-gray-700)',
                        }}
                    >
                        Exporting can take several minutes and additional memory.
                    </Text>
                    {isCompression && <Text>INT8 compresses weights, not activations.</Text>}
                    {isSubmitting && <div role='status'>Requesting export…</div>}
                    {hasSubmitted && Boolean(error) && (
                        <div role='alert'>{getApiErrorMessage(error) ?? 'Export failed. Please try again.'}</div>
                    )}
                </Flex>
            </Content>
            <ButtonGroup>
                <Button variant='secondary' isDisabled={isSubmitting} onPress={close}>
                    Cancel
                </Button>
                <Button
                    variant='accent'
                    isPending={isSubmitting}
                    isDisabled={isExportDisabled || selection === undefined}
                    onPress={() => {
                        if (selection) {
                            setHasSubmitted(true);
                            onExport(selection);
                        }
                    }}
                >
                    {isCompression ? 'Optimize to INT8' : 'Export model'}
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};
