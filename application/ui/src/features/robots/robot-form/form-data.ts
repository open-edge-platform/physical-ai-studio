import { SchemaRobotInput, SchemaRobotType } from '../robot-types';
import { buildSO101Body, type SO101FormData } from './catalog/so101';
import { buildWidowxBody, type WidowxFormData } from './catalog/widowxai';
import { buildBimanualBody, type BimanualFormData } from './catalog/widowxai-bimanual';

export type AnyRobotFormData = SO101FormData | WidowxFormData | BimanualFormData;

export type FormDataForSchema = {
    SO101_Follower: SO101FormData;
    SO101_Leader: SO101FormData;
    Trossen_WidowXAI_Follower: WidowxFormData;
    Trossen_WidowXAI_Leader: WidowxFormData;
    Trossen_Bimanual_WidowXAI_Follower: BimanualFormData;
    Trossen_Bimanual_WidowXAI_Leader: BimanualFormData;
};

export const buildRobotBody = (
    formData: AnyRobotFormData,
    schemaType: SchemaRobotType,
    robot_id: string
): SchemaRobotInput | null => {
    switch (schemaType) {
        case 'SO101_Follower':
        case 'SO101_Leader':
            return buildSO101Body(formData as SO101FormData, schemaType, robot_id);
        case 'Trossen_WidowXAI_Follower':
        case 'Trossen_WidowXAI_Leader':
            return buildWidowxBody(formData as WidowxFormData, schemaType, robot_id);
        case 'Trossen_Bimanual_WidowXAI_Follower':
        case 'Trossen_Bimanual_WidowXAI_Leader':
            return buildBimanualBody(formData as BimanualFormData, schemaType, robot_id);
        default:
            return null;
    }
};
