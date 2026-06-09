import { useState } from 'react';

import { Button, ButtonGroup, Checkbox, Content, Dialog, DialogContainer, Divider, Flex, Heading } from '@geti-ui/ui';

import { fetchClient } from '../../api/client';

export const useModelDownload = (modelId: string) => {
    const [includeSnapshot, setIncludeSnapshot] = useState(false);

    const basePath = fetchClient.PATH('/api/models/{model_id}/download', {
        params: { path: { model_id: modelId } },
    });
    const downloadUrl = includeSnapshot ? `${basePath}?include_snapshot=true` : basePath;

    return {
        includeSnapshot,
        setIncludeSnapshot,
        downloadUrl,
    };
};

export const ModelDownloadDialog = ({
    modelId,
    isOpen,
    onClose,
}: {
    modelId: string;
    isOpen: boolean;
    onClose: () => void;
}) => {
    const { includeSnapshot, setIncludeSnapshot, downloadUrl } = useModelDownload(modelId);

    const handleClose = () => {
        onClose();
    };

    return (
        <DialogContainer onDismiss={handleClose}>
            {isOpen && (
                <Dialog>
                    <Heading>Download model</Heading>
                    <Divider />

                    <Content>
                        <Flex direction='column' gap='size-200'>
                            <Checkbox isSelected={includeSnapshot} onChange={setIncludeSnapshot}>
                                Include training dataset snapshot
                            </Checkbox>
                        </Flex>
                    </Content>

                    <ButtonGroup>
                        <Button
                            href={downloadUrl}
                            target='_blank'
                            rel='noopener noreferrer'
                            variant='accent'
                            onPress={onClose}
                        >
                            Download
                        </Button>
                        <Button variant='secondary' onPress={handleClose}>
                            Close
                        </Button>
                    </ButtonGroup>
                </Dialog>
            )}
        </DialogContainer>
    );
};
