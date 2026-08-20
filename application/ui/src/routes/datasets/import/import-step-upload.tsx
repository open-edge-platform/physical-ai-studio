import { useEffect, useRef, useState } from 'react';

import {
    Button,
    ButtonGroup,
    Content,
    DropZone,
    FileTrigger,
    Flex,
    Heading,
    InlineAlert,
    ProgressCircle,
    Text,
    TextField,
    View,
} from '@geti-ui/ui';
import { useMutation } from '@tanstack/react-query';

import { fetchClient } from '../../../api/client';
import { SchemaDatasetImportJob } from '../../../api/openapi-spec';

type FormatHint = 'auto' | 'lerobot_v3';

const isAbortError = (error: unknown): boolean => {
    return error instanceof DOMException && error.name === 'AbortError';
};

interface UseDatasetUploadResult {
    upload: (file: File, formatHint: FormatHint, datasetName: string) => Promise<string | undefined>;
    abort: () => void;
    progress: number | null;
    mutation: ReturnType<
        typeof useMutation<{ id: string }, Error, { file: File; source: string; datasetName: string }>
    >;
}

export const useDatasetUpload = (projectId: string): UseDatasetUploadResult => {
    const [progress, setProgress] = useState<number | null>(null);
    const abortRef = useRef<XMLHttpRequest | null>(null);

    useEffect(() => {
        return () => {
            abortRef.current?.abort();
        };
    }, []);

    const mutation = useMutation({
        mutationFn: async ({ file, source, datasetName }: { file: File; source: string; datasetName: string }) => {
            const { data: preparedJob, error } = await fetchClient.POST(
                '/api/projects/{project_id}/imports/datasets:prepare',
                {
                    params: { path: { project_id: projectId } },
                    body: { format_hint: source, dataset_name: datasetName },
                    bodySerializer: (body) => {
                        const fd = new FormData();
                        const b = body as { format_hint: string; dataset_name: string };
                        fd.append('format_hint', b.format_hint);
                        fd.append('dataset_name', b.dataset_name);
                        return fd;
                    },
                }
            );

            if (error || !preparedJob) {
                throw new Error('Failed to prepare dataset import job');
            }

            const jobId = (preparedJob as { id?: string }).id;
            if (!jobId) {
                throw new Error('Failed to prepare import: missing job id');
            }

            const uploadPath = fetchClient.PATH('/api/projects/{project_id}/imports/datasets/{job_id}:upload', {
                params: { path: { project_id: projectId, job_id: jobId } },
            });

            return await new Promise<{ id: string }>((resolve, reject) => {
                const xhr = new XMLHttpRequest();
                abortRef.current = xhr;
                xhr.open('PUT', uploadPath);
                xhr.responseType = 'json';

                xhr.upload.onprogress = (event) => {
                    if (event.lengthComputable && event.total > 0) {
                        setProgress(Math.round((event.loaded / event.total) * 100));
                    } else {
                        setProgress(null);
                    }
                };

                xhr.onload = () => {
                    if (xhr.status >= 200 && xhr.status < 300) {
                        const uploaded = xhr.response as { id?: string } | null;
                        resolve({ id: uploaded?.id ?? jobId });
                    } else {
                        reject(new Error(`Failed to upload dataset archive: ${xhr.status}`));
                    }
                };

                xhr.onerror = () => reject(new Error('Failed to upload dataset archive'));
                xhr.onabort = () => reject(new DOMException('Upload aborted', 'AbortError'));

                const fd = new FormData();
                fd.append('archive', file);
                xhr.send(fd);
            });
        },
        onMutate: () => {
            setProgress(null);
        },
        onSettled: () => {
            abortRef.current = null;
        },
    });

    const upload = async (file: File, formatHint: FormatHint, datasetName: string): Promise<string | undefined> => {
        try {
            const job = await mutation.mutateAsync({ file, source: formatHint, datasetName });
            return job.id;
        } catch (error) {
            if (isAbortError(error)) {
                return undefined;
            }
            return undefined;
        }
    };

    const abort = () => {
        abortRef.current?.abort();
        mutation.reset();
        setProgress(null);
    };

    return { upload, abort, progress, mutation };
};

interface ImportStepUploadProps {
    importJob: SchemaDatasetImportJob | undefined;
    project_id: string;
    onClose: () => void;
    datasetName: string;
    onDatasetNameChange: (value: string) => void;
    onUploaded: (jobId: string) => void;
}

export const ImportStepUpload = ({
    importJob,
    project_id,
    onClose,
    datasetName,
    onDatasetNameChange,
    onUploaded,
}: ImportStepUploadProps) => {
    const [archive, setArchive] = useState<File | null>(null);

    const errorMessage =
        importJob?.status === 'failed'
            ? (importJob.message ?? 'Import failed during processing.')
            : importJob?.status === 'canceled'
              ? 'Import was canceled.'
              : undefined;

    const { upload, abort, progress, mutation } = useDatasetUpload(project_id);
    const canUpload = archive !== null && datasetName.trim().length > 0;

    const startUpload = async (file: File) => {
        const jobId = await upload(file, 'auto', datasetName.trim());
        if (jobId) {
            onUploaded(jobId);
        }
    };

    const onUpload = async () => {
        if (archive === null || !canUpload) {
            return;
        }
        await startUpload(archive);
    };

    const onFileSelected = (file: File) => {
        setArchive(file);
        if (datasetName.trim().length === 0) {
            const suggestion = file.name.endsWith('.zip') ? file.name.slice(0, -4) : file.name;
            onDatasetNameChange(suggestion);
        }
    };

    const handleFileList = (files: FileList | null) => {
        const file = files?.[0] ?? null;
        if (file === null) {
            return;
        }

        onFileSelected(file);
    };

    const onCancel = () => {
        if (mutation.isPending) {
            abort();
            return;
        }
        onClose();
    };

    return (
        <>
            <Content>
                {errorMessage ? (
                    <InlineAlert variant='negative'>
                        <Heading>Import setup error</Heading>
                        <Content>{errorMessage}</Content>
                    </InlineAlert>
                ) : null}

                <Flex direction='column' gap='size-200'>
                    <TextField
                        isRequired
                        width='100%'
                        label='Dataset name'
                        value={datasetName}
                        onChange={onDatasetNameChange}
                        isDisabled={mutation.isPending}
                    />

                    <DropZone
                        isFilled={archive !== null}
                        onDrop={async (e) => {
                            if (mutation.isPending) {
                                return;
                            }
                            const fileItem = e.items.find((item) => item.kind === 'file');
                            if (fileItem?.kind !== 'file') {
                                return;
                            }
                            const file = await fileItem.getFile();
                            if (file.name.endsWith('.zip')) {
                                onFileSelected(file);
                            }
                        }}
                    >
                        {mutation.isPending ? (
                            <Flex direction='column' alignItems='center' justifyContent='center' gap='size-100'>
                                {progress !== null && (
                                    <ProgressCircle
                                        value={progress}
                                        minValue={0}
                                        maxValue={100}
                                        size='M'
                                        aria-label='Uploading dataset archive'
                                    />
                                )}
                                <Text>{progress === null ? 'Uploading dataset archive...' : `${progress}%`}</Text>
                            </Flex>
                        ) : (
                            <Flex direction='column' gap='size-100'>
                                {archive !== null ? (
                                    <Text>{archive.name}</Text>
                                ) : (
                                    <Text>Drop a .zip archive here or click to browse</Text>
                                )}
                                <View>
                                    <FileTrigger acceptedFileTypes={['.zip']} onSelect={handleFileList}>
                                        <Button variant='secondary'>
                                            {archive !== null ? 'Choose a different file' : 'Browse'}
                                        </Button>
                                    </FileTrigger>
                                </View>
                            </Flex>
                        )}
                    </DropZone>
                </Flex>

                {mutation.isError && !isAbortError(mutation.error) ? (
                    <InlineAlert variant='negative'>
                        <Heading>Import error</Heading>
                        <Content>Failed to upload dataset archive. Please try again.</Content>
                    </InlineAlert>
                ) : null}
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={onCancel}>
                    {mutation.isPending ? 'Abort upload' : 'Cancel'}
                </Button>
                <Button
                    variant='accent'
                    onPress={onUpload}
                    isPending={mutation.isPending}
                    isDisabled={!canUpload || mutation.isPending}
                >
                    Upload
                </Button>
            </ButtonGroup>
        </>
    );
};
