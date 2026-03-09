import { useParams } from 'react-router';
import { defaultBackend } from '../../../features/configuration/shared/backend-selection';

export const useInferenceParams = () => {
    const { project_id, model_id, backend } = useParams();

    if (project_id === undefined) {
        throw new Error('Unknown project_id parameter');
    }

    if (model_id === undefined) {
        throw new Error('Unknown model_id parameter');
    }


    return { project_id, model_id, backend: backend ?? defaultBackend };
};
