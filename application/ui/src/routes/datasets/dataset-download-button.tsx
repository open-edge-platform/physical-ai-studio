import { useRef, useState } from 'react';

import { Button, ButtonGroup, Content, Dialog, DialogTrigger, Divider, Heading, Text } from '@geti/ui';
import { useMutation } from '@tanstack/react-query';

import { fetchClient } from '../../api/client';
import { DownloadProgressContent } from '../../components/download-progress-content';
import { getArchiveBlobFromResponse, getFilenameFromContentDisposition, isAbortError } from '../utils/download';

const useDatasetDownload = (datasetId: string) => {
    const [isDialogOpen, setDialogOpen] = useState(false);
    const [progress, setProgress] = useState<number | null>(null);

    const abortRef = useRef<AbortController | null>(null);

    const downloadPath = fetchClient.PATH('/api/dataset/{dataset_id}/download', {
        params: { path: { dataset_id: datasetId } },
    });

    const mutation = useMutation({
        mutationFn: async () => {
            const abortController = new AbortController();
            abortRef.current = abortController;

            const response = await fetch(downloadPath, {
                signal: abortController.signal,
            });

            if (!response.ok) {
                throw new Error(`Failed to export dataset: ${response.status}`);
            }

            const filename = getFilenameFromContentDisposition(
                response.headers.get('content-disposition'),
                'dataset.zip'
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
            setDialogOpen(true);
            setProgress(null);
        },
        onError: (error) => {
            if (isAbortError(error)) {
                setDialogOpen(false);
            }
        },
        onSuccess: () => {
            setDialogOpen(false);
        },
        onSettled: () => {
            abortRef.current = null;
        },
    });

    const cancelDownload = () => {
        abortRef.current?.abort();
    };

    const closeDialog = () => {
        if (mutation.isPending) {
            cancelDownload();
        }

        setDialogOpen(false);
    };

    return {
        mutation,
        progress,
        isDialogOpen,
        cancelDownload,
        closeDialog,
    };
};

export const DatasetDownloadButton = ({ datasetId }: { datasetId: string }) => {
    const { mutation, progress, isDialogOpen, closeDialog, cancelDownload } = useDatasetDownload(datasetId);

    return (
        <DialogTrigger isOpen={isDialogOpen} onOpenChange={(open) => !open && closeDialog()}>
            <Button
                variant='secondary'
                alignSelf={'center'}
                isPending={mutation.isPending}
                onPress={() => mutation.mutate()}
            >
                <Text>Download dataset</Text>
            </Button>

            <Dialog>
                <Heading>Downloading dataset</Heading>
                <Divider />

                <Content>
                    <DownloadProgressContent
                        isError={mutation.isError}
                        isPending={mutation.isPending}
                        progress={progress}
                        errorMessage='Failed to download dataset. Please try again.'
                        preparingMessage='Preparing export and starting download…'
                    />
                </Content>

                <ButtonGroup>
                    <Button variant='secondary' onPress={mutation.isPending ? cancelDownload : closeDialog}>
                        {mutation.isPending ? 'Cancel' : 'Close'}
                    </Button>
                </ButtonGroup>
            </Dialog>
        </DialogTrigger>
    );
};
