import { useState } from 'react';

import { Button, ButtonGroup, Content, Dialog, Divider, Flex, Heading, RadioGroup, Text } from '@geti-ui/ui';

import { getApiErrorMessage } from '../../../api/errors';
import type { SchemaModelExportAction } from '../../../api/openapi-spec';
import { BackendOption } from '../backend-selection/backend-option';

type ExportBackend = SchemaModelExportAction['backend'];

interface AddVariantDialogProps {
    backends: ExportBackend[];
    isSubmitting: boolean;
    isExportDisabled: boolean;
    error: unknown;
    onExport: (backend: ExportBackend) => void;
    close: () => void;
}

export const AddVariantDialog = ({
    backends,
    isSubmitting,
    isExportDisabled,
    error,
    onExport,
    close,
}: AddVariantDialogProps) => {
    const [selectedBackend, setSelectedBackend] = useState(backends[0]);
    const [hasSubmitted, setHasSubmitted] = useState(false);

    return (
        <Dialog>
            <Heading>Add variant</Heading>
            <Divider />
            <Content>
                <Flex direction='column' gap='size-200'>
                    <Text>
                        Select a backend to export the stored training checkpoint. Existing variants stay unchanged.
                    </Text>
                    <RadioGroup
                        label='Export backend'
                        value={selectedBackend}
                        onChange={(value) => {
                            const backend = backends.find((option) => option === value);
                            if (backend) {
                                setSelectedBackend(backend);
                                setHasSubmitted(false);
                            }
                        }}
                        isDisabled={isExportDisabled}
                        isEmphasized
                        width='100%'
                    >
                        <Flex direction='column' gap='size-200'>
                            {backends.map((backend) => (
                                <BackendOption key={backend} id={backend} isSelected={backend === selectedBackend} />
                            ))}
                        </Flex>
                    </RadioGroup>
                    <Text>
                        Exporting can take several minutes and additional memory. Progress will appear on the variant
                        card.
                    </Text>
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
                    isDisabled={isExportDisabled || selectedBackend === undefined}
                    onPress={() => {
                        if (selectedBackend) {
                            setHasSubmitted(true);
                            onExport(selectedBackend);
                        }
                    }}
                >
                    Export variant
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};
