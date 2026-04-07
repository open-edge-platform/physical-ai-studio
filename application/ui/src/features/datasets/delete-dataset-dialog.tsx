import { useState } from 'react';

import { AlertDialog, Checkbox, Flex, Text } from '@geti-ui/ui';

import { $api } from '../../api/client';
import { SchemaDatasetOutput } from '../../api/openapi-spec';

type Dataset = SchemaDatasetOutput;

export const DeleteDatasetDialog = ({ dataset, onDone }: { dataset: Dataset; onDone: () => void }) => {
    const [removeFiles, setRemoveFiles] = useState(false);
    const deleteMutation = $api.useMutation('delete', '/api/dataset/{dataset_id}');

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
                <Checkbox isSelected={removeFiles} onChange={setRemoveFiles}>
                    Also remove dataset files from disk
                </Checkbox>
            </Flex>
        </AlertDialog>
    );
};
