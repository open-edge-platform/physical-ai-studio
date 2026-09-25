import { useState } from 'react';

import { AlertDialog, Checkbox, Flex, Text } from '@geti-ui/ui';
import { isEmpty } from 'lodash-es';

import { $api } from '../../api/client';
import { SchemaDatasetOutput } from '../../api/openapi-spec';
import { useProjectId } from '../projects/use-project';

type Dataset = SchemaDatasetOutput;

const useEpisodes = (datasetId: string | undefined) => {
    return $api.useQuery(
        'get',
        '/api/dataset/{dataset_id}/episodes',
        {
            params: {
                path: {
                    dataset_id: String(datasetId),
                },
            },
        },
        {
            enabled: datasetId !== undefined,
        }
    );
};

export const DeleteDatasetDialog = ({ dataset, onDone }: { dataset: Dataset; onDone: () => void }) => {
    const [removeFiles, setRemoveFiles] = useState(false);
    const { project_id } = useProjectId();
    const { data: episodes } = useEpisodes(dataset.id);
    const deleteMutation = $api.useMutation('delete', '/api/dataset/{dataset_id}', {
        meta: {
            invalidates: [
                ['get', '/api/projects'],
                ['get', '/api/projects/{project_id}', { params: { path: { project_id } } }],
            ],
        },
    });

    const onDelete = async () => {
        if (dataset.id === undefined) {
            return;
        }

        await deleteMutation.mutateAsync({
            params: {
                path: {
                    dataset_id: dataset.id,
                },
                query: {
                    remove_files: removeFiles,
                },
            },
        });

        onDone();
    };

    return (
        <AlertDialog title='Delete dataset' variant='warning' primaryActionLabel='Delete' onPrimaryAction={onDelete}>
            <Flex direction='column' gap='size-200'>
                <Text>Are you sure you want to delete this dataset?</Text>
                {!isEmpty(episodes) && (
                    <Checkbox isEmphasized isSelected={removeFiles} onChange={setRemoveFiles}>
                        Also remove dataset files from disk
                    </Checkbox>
                )}
            </Flex>
        </AlertDialog>
    );
};
