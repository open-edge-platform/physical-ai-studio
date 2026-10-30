import { useEffect } from 'react';

import { useRobotCatalogDefinitionQuery } from './robot-catalog.hooks';
import { mapJointToURDFJoint, useLoadModelQuery } from './robot-models-context';
import { SchemaRobotType } from './robot-types';
import { JointsState } from './use-robot-observations';

export const useSynchronizeModelJoints = (joints: JointsState, robotType: SchemaRobotType) => {
    const { data: definition } = useRobotCatalogDefinitionQuery(robotType);
    const jointMap = definition.joint_map;

    const { data: model } = useLoadModelQuery(robotType);

    useEffect(() => {
        if (!model) return;

        joints.forEach((joint) => {
            mapJointToURDFJoint(joint, model, jointMap);
        });
    }, [model, joints, jointMap]);
};
