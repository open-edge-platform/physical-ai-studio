import type { SchemaRobotInput } from '../robot-types';
import type { RobotPayload } from './provider';

export const buildRobotBody = (
    formData: { name: string; payload: RobotPayload },
    schemaType: string,
    robot_id: string
): SchemaRobotInput | null => {
    if (formData.name.trim() === '') return null;
    return {
        id: robot_id,
        name: formData.name,
        type: schemaType,
        payload: formData.payload,
    } as unknown as SchemaRobotInput;
};
