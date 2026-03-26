import { useRef, useState } from 'react';

import { Button, ButtonGroup, Checkbox, Content, Dialog, DialogContainer, Divider, Flex, Heading } from '@geti/ui';
import { useMutation } from '@tanstack/react-query';

import { fetchClient } from '../../api/client';
import { DownloadProgressContent } from '../../components/download-progress-content';
import { getArchiveBlobFromResponse, getFilenameFromContentDisposition, isAbortError } from '../utils/download';

export const useModelDownload = (modelId: string) => {
    const [progress, setProgress] = useState<number | null>(null);
    const [includeSnapshot, setIncludeSnapshot] = useState(false);

    const abortRef = useRef<AbortController | null>(null);

    const mutation = useMutation({
        mutationFn: async ({ includeSnapshot: withSnapshot }: { includeSnapshot: boolean }) => {
            const abortController = new AbortController();
            abortRef.current = abortController;

            const basePath = fetchClient.PATH('/api/models/{model_id}/download', {
                params: { path: { model_id: modelId } },
            });

            const downloadUrl = withSnapshot ? `${basePath}?include_snapshot=true` : basePath;

            const response = await fetch(downloadUrl, {
                signal: abortController.signal,
            });

            if (!response.ok) {
                throw new Error(`Failed to export model: ${response.status}`);
            }

            const filename = getFilenameFromContentDisposition(
                response.headers.get('content-disposition'),
                'model.zip'
            );

            const archiveBlob = await getArchiveBlobFromResponse(response, setProgress);

            const blobUrl = URL.createObjectURL(archiveBlob);

            const link = document.createElement('a');
            link.href = blobUrl;
            link.download = filename;

            document.body.appendChild(link);
            link.click();
            link.remove();

            URL.revokeObjectURL(blobUrl);
        },
        onMutate: () => {
            setProgress(null);
        },
        onSettled: () => {
            abortRef.current = null;
        },
    });

    const cancelDownload = () => {
        abortRef.current?.abort();
    };

    return {
        mutation,
        progress,
        includeSnapshot,
        setIncludeSnapshot,
        cancelDownload,
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
    const { mutation, progress, includeSnapshot, setIncludeSnapshot, cancelDownload } = useModelDownload(modelId);

    const handleClose = () => {
        if (mutation.isPending) {
            cancelDownload();
        }
        mutation.reset();
        onClose();
    };

    const handleDownload = () => {
        mutation.mutate(
            { includeSnapshot },
            {
                onSuccess: () => {
                    onClose();
                },
                onError: (error) => {
                    if (isAbortError(error)) {
                        onClose();
                    }
                },
            }
        );
    };

    return (
        <DialogContainer onDismiss={handleClose}>
            {isOpen && (
                <Dialog>
                    <Heading>Download model</Heading>
                    <Divider />

                    <Content>
                        {mutation.isPending || mutation.isError ? (
                            <DownloadProgressContent
                                isError={mutation.isError}
                                isPending={mutation.isPending}
                                progress={progress}
                                errorMessage='Failed to download model. Please try again.'
                                preparingMessage='Preparing export and starting download...'
                            />
                        ) : (
                            <Flex direction='column' gap='size-200'>
                                <Checkbox isSelected={includeSnapshot} onChange={setIncludeSnapshot}>
                                    Include training dataset snapshot
                                </Checkbox>
                            </Flex>
                        )}
                    </Content>

                    <ButtonGroup>
                        {!mutation.isPending && !mutation.isError && (
                            <Button variant='accent' onPress={handleDownload}>
                                Download
                            </Button>
                        )}
                        <Button variant='secondary' onPress={mutation.isPending ? cancelDownload : handleClose}>
                            {mutation.isPending ? 'Cancel' : 'Close'}
                        </Button>
                    </ButtonGroup>
                </Dialog>
            )}
        </DialogContainer>
    );
};
