import {
    Button,
    ButtonGroup,
    Content,
    Flex,
    Heading,
    InlineAlert,
    Item,
    Picker,
    Text,
    TextField,
    View,
} from '@geti-ui/ui';
import { isNumber } from 'lodash-es';

import { $api } from '../../../api/client';
import { SchemaDatasetImportJob, SchemaManifestCameraEntry, SchemaManifestRobotEntry } from '../../../api/openapi-spec';
import { formatDuration } from '../../models/utils';
import type { FinalizeFields } from './use-dataset-import-job-state';

const RobotsSummary = ({ robots }: { robots: Array<SchemaManifestRobotEntry> }) => {
    return (
        <Flex direction='column' gap='size-50'>
            <Text>
                <strong>Robots</strong>
            </Text>
            {robots.map((robot, index) => (
                <Text key={`${robot.type ?? 'robot'}-${index}`}>
                    {robot.type ?? `Robot ${index + 1}`} with <strong>{robot.joints?.length ?? 0}</strong>
                    joints
                </Text>
            ))}
        </Flex>
    );
};

const CamerasSummary = ({ cameras }: { cameras: Array<SchemaManifestCameraEntry> }) => {
    return (
        <Flex direction='column' gap='size-50'>
            <Text>
                <strong>Cameras</strong>
            </Text>
            {cameras.map((camera, index) => (
                <Text key={`${camera.name ?? 'camera'}-${index}`}>
                    {camera.name ?? `Camera ${index + 1}`}:{' '}
                    <strong>
                        {camera.width ?? '—'}×{camera.height ?? '—'}
                    </strong>{' '}
                    @ <strong>{camera.fps ?? '-'}</strong> FPS
                </Text>
            ))}
        </Flex>
    );
};

const DatasetManifestSummary = ({ importJob }: { importJob: SchemaDatasetImportJob }) => {
    const importPayload = importJob?.payload;
    const draft = importPayload?.dataset_manifest_draft;

    if (draft === undefined || draft === null) {
        return null;
    }

    const cameras = draft.dataset_schema?.cameras ?? [];
    const robots = draft.dataset_schema?.robots ?? [];

    const fps = draft.dataset_schema?.cameras?.at(0)?.fps;
    const episodeCount = draft.statistics?.episode_count;
    const frameCount = draft.statistics?.frame_count;
    const time = isNumber(fps) && fps > 0 ? Number(frameCount) / fps : null;

    return (
        <Flex direction='column' gap='size-150'>
            <Heading level={4}>Dataset import summary</Heading>
            <Text>
                Found <strong>{episodeCount}</strong> episodes{' '}
                {time !== null ? <>with total length of {formatDuration(1000 * time)}</> : null}.
            </Text>

            <Flex direction='column' gap='size-100'>
                {robots.length > 0 && <RobotsSummary robots={robots} />}

                {cameras.length > 0 && <CamerasSummary cameras={cameras} />}
            </Flex>
        </Flex>
    );
};

interface ImportStepUserReviewProps {
    importJob: SchemaDatasetImportJob;
    project_id: string;
    onClose: () => void;
    fields: FinalizeFields;
    onFieldsChange: (fields: FinalizeFields) => void;
}

export const ImportStepUserReview = ({
    importJob,
    project_id,
    onClose,
    fields,
    onFieldsChange,
}: ImportStepUserReviewProps) => {
    const { data: environments } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/environments', {
        params: { path: { project_id } },
    });

    const finalizeMutation = $api.useMutation('post', '/api/projects/{project_id}/imports/datasets/{job_id}:finalize', {
        meta: {
            invalidates: [['get', '/api/jobs/{job_id}', { params: { path: { job_id: importJob.id! } } }]],
        },
    });
    const canFinalize = fields.environmentId !== undefined;

    const onFinalize = () => {
        if (!canFinalize || fields.environmentId === undefined) {
            return;
        }

        finalizeMutation.mutate({
            params: {
                path: { project_id, job_id: importJob.id! },
            },
            body: {
                environment_id: fields.environmentId,
                default_task: fields.defaultTask,
            },
        });
    };

    return (
        <>
            <Content>
                <View
                    backgroundColor='gray-50'
                    borderColor='gray-200'
                    borderWidth='thick'
                    padding='size-200'
                    marginBottom='size-200'
                >
                    <DatasetManifestSummary importJob={importJob} />
                </View>

                <View marginY='size-100'>
                    <Text>
                        Analysis complete. Review the metadata above, then provide environment and optional task to
                        finalize the import.
                    </Text>
                    {finalizeMutation.isError ? (
                        <InlineAlert variant='negative'>
                            <Heading>Finalize import failed</Heading>
                            <Content>Failed to finalize import. Please check job state and try again.</Content>
                        </InlineAlert>
                    ) : null}
                </View>

                <Flex direction='column' gap='size-100'>
                    <Picker
                        label='Recording environment'
                        width='100%'
                        items={environments}
                        selectedKey={fields.environmentId}
                        onSelectionChange={(value) =>
                            onFieldsChange({
                                ...fields,
                                environmentId: value === null ? undefined : value.toString(),
                            })
                        }
                    >
                        {(item) => <Item key={item.id}>{item.name}</Item>}
                    </Picker>

                    <TextField
                        width='100%'
                        label='Task'
                        value={fields.defaultTask}
                        onChange={(value) => onFieldsChange({ ...fields, defaultTask: value })}
                    />
                </Flex>
            </Content>

            <ButtonGroup>
                <Button variant='secondary' onPress={onClose} isDisabled={finalizeMutation.isPending}>
                    Cancel
                </Button>
                <Button
                    variant='accent'
                    onPress={onFinalize}
                    isPending={finalizeMutation.isPending}
                    isDisabled={!canFinalize}
                >
                    Finalize import
                </Button>
            </ButtonGroup>
        </>
    );
};
