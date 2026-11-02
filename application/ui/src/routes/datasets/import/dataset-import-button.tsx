import { Button, DialogTrigger, Text } from '@geti-ui/ui';

import { ImportDatasetDialog } from './import-dataset-dialog';

interface DatasetImportButtonProps {
    existingJobId?: string;
    buttonLabel?: string;
    onPendingJobDismissed?: (jobId: string) => void;
    onImportCompleted?: (datasetId: string) => void;
}

export const DatasetImportButton = ({
    existingJobId,
    buttonLabel = 'Import dataset',
    onPendingJobDismissed,
    onImportCompleted,
}: DatasetImportButtonProps = {}) => {
    return (
        <DialogTrigger>
            <Button variant='secondary' alignSelf={'center'}>
                <Text>{buttonLabel}</Text>
            </Button>

            {(close) => (
                <ImportDatasetDialog
                    initialJobId={existingJobId}
                    onPendingJobDismissed={onPendingJobDismissed}
                    onImportCompleted={onImportCompleted}
                    onClose={close}
                />
            )}
        </DialogTrigger>
    );
};

export { ImportDatasetDialog } from './import-dataset-dialog';
