import { useParams } from 'react-router';

export function useDatasetId() {
    const { project_id, dataset_id } = useParams<{ project_id: string; dataset_id: string }>();

    if (project_id === undefined) {
        throw new Error('Unkown project_id parameter');
    }
    if (dataset_id === undefined) {
        throw new Error('Unkown dataset_id parameter');
    }

    return { project_id, dataset_id };
}
