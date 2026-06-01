import { useParams } from 'react-router';
import { useSearchParams } from 'react-router-dom';

import { defaultBackend } from '../../../features/models/backend-selection';

export const useInferenceParams = () => {
    const { project_id, model_id, backend } = useParams();
    const [searchParams] = useSearchParams();

    if (project_id === undefined) {
        throw new Error('Unknown project_id parameter');
    }

    if (model_id === undefined) {
        throw new Error('Unknown model_id parameter');
    }

    return {
        project_id,
        model_id,
        backend: backend ?? defaultBackend,
        device: searchParams.get('device') ?? undefined,
    };
};
