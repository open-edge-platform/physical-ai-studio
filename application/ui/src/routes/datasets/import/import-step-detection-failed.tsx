import { Button, ButtonGroup, Content, Flex, Text, View } from '@geti-ui/ui';

import { SchemaDatasetImportJob } from '../../../api/openapi-spec';

const getValidationErrors = (messages: { severity: string; message: string }[] | undefined) => {
    return (messages ?? [])
        .filter((m) => m.severity === 'error')
        .map((m) => m.message)
        .filter(Boolean);
};

interface ImportStepDetectionFailedProps {
    importJob: SchemaDatasetImportJob;
    onClose: () => void;
}

export const ImportStepDetectionFailed = ({ importJob, onClose }: ImportStepDetectionFailedProps) => {
    const importPayload = importJob?.payload;
    const validationErrors = getValidationErrors(importPayload?.validation_report?.messages);

    const usedAutoDetection = importPayload?.format_hint === 'auto' || importPayload?.format_hint === undefined;

    const errorMessage = usedAutoDetection
        ? 'Could not automatically detect the dataset format. Please select a format manually.'
        : 'Could not detect the selected dataset format. Please choose a different format and retry.';

    const onCancel = () => {
        onClose();
    };

    return (
        <>
            <Content>
                <Flex direction='column' gap='size-100'>
                    {validationErrors.length > 0 ? (
                        <View backgroundColor='gray-75' borderColor='gray-200' borderWidth='thin' padding='size-150'>
                            <Flex direction='column' gap='size-50'>
                                <Text>
                                    <strong>Detection details</strong>
                                </Text>
                                <ul>
                                    {validationErrors.map((message, index) => (
                                        <li key={`${message}-${index}`}>{message}</li>
                                    ))}
                                </ul>
                            </Flex>
                        </View>
                    ) : null}

                    {validationErrors.length === 0 ? <Text>{errorMessage}</Text> : null}

                    <Text>Please check the application logs for more details or raise an issue on our Github.</Text>
                </Flex>
            </Content>

            <ButtonGroup>
                <Button variant='secondary' onPress={onCancel}>
                    Cancel
                </Button>
            </ButtonGroup>
        </>
    );
};
