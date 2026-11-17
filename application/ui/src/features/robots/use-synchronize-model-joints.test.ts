import { describe, expect, it, rs } from '@rstest/core';
import { degToRad } from 'three/src/math/MathUtils.js';
import { URDFJoint, URDFRobot } from 'urdf-loader/src/URDFClasses.js';

import { renderHook } from '../../test-utils/render';
import * as actualRobotModels from './robot-models-context' with { rstest: 'importActual' };
import { RobotModelConfig } from './robot-models-context';
import { useSynchronizeModelJoints } from './use-synchronize-model-joints';

const { model, jointMap } = rs.hoisted(() => ({
    model: { current: undefined as URDFRobot | undefined },
    jointMap: { 'shoulder_pan.pos': ['arm'] },
}));

rs.mock('./robot-catalog.hooks', () => ({
    useRobotCatalogDefinitionQuery: () => ({ data: { joint_map: jointMap } }),
}));

rs.mock('./robot-models-context', () => ({
    ...actualRobotModels,
    useLoadModelQuery: () => ({ data: model.current }),
}));

describe('useSynchronizeModelJoints', () => {
    it('only remaps joints when the mapping inputs change, not the robot object identity', () => {
        const robotModel = new URDFRobot();
        const joint = new URDFJoint();
        joint.jointType = 'revolute';
        joint.limit.lower = -1;
        joint.limit.upper = 2;
        robotModel.joints = { arm: joint };
        model.current = robotModel;
        const setJointValue = rs.spyOn(robotModel, 'setJointValue');
        const joints = [{ name: 'shoulder_pan.pos', value: 45 }];
        let robot: RobotModelConfig = { type: 'SO101_Follower', payload: { unit: 'normalized' } };

        const { rerender } = renderHook(() => useSynchronizeModelJoints(joints, robot));
        expect(setJointValue).toHaveBeenCalledTimes(1);

        robot = { type: 'SO101_Follower', payload: { unit: 'normalized', serial_number: 'updated' } };
        rerender();
        expect(setJointValue).toHaveBeenCalledTimes(1);

        robot = { type: 'SO101_Follower', payload: { unit: 'degrees' } };
        rerender();
        expect(setJointValue).toHaveBeenCalledTimes(2);
        expect(joint.angle).toBeCloseTo(degToRad(45));

        robot = { type: 'SO101_Leader', payload: { unit: 'degrees' } };
        rerender();
        expect(setJointValue).toHaveBeenCalledTimes(3);

        robot = { type: 'SO101_Follower' };
        rerender();
        expect(setJointValue).toHaveBeenCalledTimes(4);

        robot = { type: 'SO101_Follower' };
        rerender();
        expect(setJointValue).toHaveBeenCalledTimes(4);
    });
});
