import type { SchemaTrossenSingleArmPayload } from '../../../../api/openapi-spec';
import type { SchemaRobotInput, SchemaRobotType } from '../../robot-types';

export const buildWidowxBody = (
    formData: {
        name: string;
        payload: SchemaTrossenSingleArmPayload;
    },
    schemaType: SchemaRobotType,
    robot_id: string
): SchemaRobotInput | null => {
    if (!formData.payload.connection_string) {
        return null;
    }

    return {
        id: robot_id,
        name: formData.name,
        type: schemaType,
        payload: formData.payload,
    } as SchemaRobotInput;
};
