import { useMemo, useRef, useState } from 'react';

import { Button, ButtonGroup, Content, Dialog, Divider, Heading } from '@geti-ui/ui';
import { useQuery } from '@tanstack/react-query';

import { fetchClient } from '../../api/client';
import { DownloadProgressContent } from '../../components/download-progress-content';
import {
    getArchiveBlobFromResponse,
    getFilenameFromContentDisposition,
    isAbortError,
} from '../../routes/utils/download';

const useDatasetDownload = (datasetId: string) => {
    const [progress, setProgress] = useState<number | null>(null);
    const abortRef = useRef<AbortController | null>(null);

    const queryKey = useMemo(() => ['get', '/api/dataset/{dataset_id}/download', datasetId] as const, [datasetId]);

    const downloadPath = fetchClient.PATH('/api/dataset/{dataset_id}/download', {
        params: { path: { dataset_id: datasetId } },
    });

    const query = useQuery({
        queryKey,
        retry: false,
        refetchOnWindowFocus: false,
        refetchOnReconnect: false,
        queryFn: async () => {
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
            abortRef.current = null;
            return { success: true };
        },
    });

    const cancelDownload = () => {
        abortRef.current?.abort();
    };

    return {
        query,
        progress,
        cancelDownload,
    };
};

export const DatasetDownloadDialog = ({
    datasetId,
    onCloseDialog,
}: {
    datasetId: string;
    onCloseDialog: () => void;
}) => {
    const { query, progress, cancelDownload } = useDatasetDownload(datasetId);

    const isCancellationError =
        query.isError &&
        (isAbortError(query.error) ||
            (typeof query.error === 'object' &&
                query.error !== null &&
                'name' in query.error &&
                (query.error as { name?: string }).name === 'CanceledError'));

    const handleClose = () => {
        if (query.isFetching) {
            cancelDownload();
            return;
        }

        onCloseDialog();
    };

    return (
        <Dialog>
            <Heading>Downloading dataset</Heading>
            <Divider />

            <Content>
                <DownloadProgressContent
                    isError={query.isError && !query.isFetching && !isCancellationError}
                    isPending={query.isFetching}
                    progress={progress}
                    errorMessage='Failed to download dataset. Please try again.'
                    preparingMessage='Preparing export and starting download…'
                />
            </Content>

            <ButtonGroup>
                <Button variant='secondary' onPress={handleClose}>
                    {query.isFetching ? 'Cancel' : 'Close'}
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};
