import { useRef, useState } from 'react';

import {
    Button,
    ButtonGroup,
    Content,
    Dialog,
    DialogTrigger,
    Divider,
    Flex,
    Heading,
    ProgressBar,
    ProgressCircle,
    Text,
} from '@geti/ui';
import { useMutation } from '@tanstack/react-query';

import { fetchClient } from '../../api/client';

const getFilenameFromContentDisposition = (contentDisposition: string | null): string => {
    if (!contentDisposition) return 'dataset.zip';

    const utfFilenameMatch = contentDisposition.match(/filename\*=UTF-8''([^;]+)/i);
    if (utfFilenameMatch?.[1]) {
        return decodeURIComponent(utfFilenameMatch[1]);
    }

    const filenameMatch = contentDisposition.match(/filename="?([^";]+)"?/i);
    return filenameMatch?.[1] ?? 'dataset.zip';
};

const isAbortError = (error: unknown): boolean => {
    return error instanceof DOMException && error.name === 'AbortError';
};

/**
 * Reads a streaming HTTP response and reconstructs the final archive Blob.
 *
 * Instead of calling `response.blob()`, we manually read the stream using a
 * `ReadableStreamDefaultReader`. This allows us to:
 *
 * - Track download progress when the `content-length` header is available
 * - Avoid buffering the entire file before we can show progress
 *
 * The browser fetch API exposes the response body as a `ReadableStream`.
 * We repeatedly read chunks from the stream until `done === true`, while
 * accumulating them in memory.
 */
const getArchiveBlobFromResponse = async (
    response: Response,
    onProgress: (progress: number | null) => void
): Promise<Blob> => {
    const contentLength = response.headers.get('content-length');
    const totalBytes = contentLength ? Number(contentLength) : null;

    if (!response.body) {
        return response.blob();
    }

    const reader = response.body.getReader();
    const chunks: Uint8Array[] = [];

    let downloadedBytes = 0;

    while (true) {
        const { done, value } = await reader.read();

        if (done) break;

        if (value) {
            chunks.push(value);
            downloadedBytes += value.length;

            if (totalBytes && Number.isFinite(totalBytes)) {
                const percent = Math.round((downloadedBytes / totalBytes) * 100);
                onProgress(percent);
            }
        }
    }

    return new Blob(chunks, { type: 'application/zip' });
};

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

            const filename = getFilenameFromContentDisposition(response.headers.get('content-disposition'));

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
                    {mutation.isError ? (
                        <Text>Failed to download dataset. Please try again.</Text>
                    ) : (
                        <Flex direction='column' gap='size-200'>
                            {progress === null ? (
                                <Flex alignItems='center' gap='size-100'>
                                    <ProgressCircle isIndeterminate size='S' />
                                    <Text>Preparing export and starting download…</Text>
                                </Flex>
                            ) : (
                                <Flex direction='column' gap='size-100'>
                                    <Text>{progress}%</Text>
                                    <ProgressBar value={progress} width='100%' />
                                </Flex>
                            )}
                        </Flex>
                    )}
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
