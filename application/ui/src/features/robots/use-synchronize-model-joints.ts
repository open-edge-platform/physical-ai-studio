import { useEffect, useMemo } from 'react';

import { useRobotCatalogDefinitionQuery } from './robot-catalog.hooks';
import { mapJointToURDFJoint, RobotModelConfig, useLoadModelQuery } from './robot-models-context';
import { JointsState } from './use-robot-observations';

export const useSynchronizeModelJoints = (joints: JointsState, robot: RobotModelConfig) => {
    const { type, payload } = robot;
    const unit = payload && 'unit' in payload ? payload.unit : undefined;
    const modelConfig = useMemo(() => ({ type, payload: { unit } }), [type, unit]);
    const { data: definition } = useRobotCatalogDefinitionQuery(type);
    const jointMap = definition.joint_map;

    const { data: model } = useLoadModelQuery(type);

    useEffect(() => {
        if (!model) return;

        joints.forEach((joint) => {
            mapJointToURDFJoint(joint, model, jointMap, modelConfig);
        });
    }, [model, joints, jointMap, modelConfig]);
};
