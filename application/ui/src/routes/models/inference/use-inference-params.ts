import { useParams } from "react-router";

export const useInferenceParams = () => {
    const { project_id, model_id } = useParams();

    if (project_id === undefined) {
        throw new Error('Unknown project_id parameter');
    }

    if (model_id === undefined) {
        throw new Error('Unknown model_id parameter');
    }

    return { project_id, model_id };
}
