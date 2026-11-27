import { describe, expect, it } from '@rstest/core';
import { degToRad } from 'three/src/math/MathUtils.js';
import { URDFJoint, URDFRobot } from 'urdf-loader/src/URDFClasses.js';

import { featureRange, mapJointToURDFJoint } from './robot-models-context';

const createModel = (jointType: URDFJoint['jointType'] = 'revolute') => {
    const model = new URDFRobot();
    const joint = new URDFJoint();
    joint.jointType = jointType;
    joint.limit.lower = -1;
    joint.limit.upper = 2;
    model.joints = { arm: joint };
    return model;
};

describe('mapJointToURDFJoint', () => {
    it.each(['SO101_Follower', 'SO101_Leader'])('maps normalized %s body joints to URDF limits', (type) => {
        const model = createModel();

        for (const [value, expected] of [
            [-100, -1],
            [0, 0.5],
            [100, 2],
            [-200, -1],
            [200, 2],
        ]) {
            mapJointToURDFJoint(
                { name: 'shoulder_pan.pos', value },
                model,
                { 'shoulder_pan.pos': ['arm'] },
                {
                    type,
                    payload: { unit: 'normalized' },
                }
            );

            expect(model.joints.arm.angle).toBeCloseTo(expected);
        }
    });

    it('maps the normalized gripper from [0, 100] rather than [-100, 100]', () => {
        const model = createModel();

        for (const [value, expected] of [
            [0, -1],
            [50, 0.5],
            [100, 2],
        ]) {
            mapJointToURDFJoint(
                { name: 'gripper.pos', value },
                model,
                { 'gripper.pos': ['arm'] },
                {
                    type: 'SO101_Follower',
                    payload: { unit: 'normalized' },
                }
            );

            expect(model.joints.arm.angle).toBeCloseTo(expected);
        }
    });

    it.each([
        { type: 'SO101_Follower', payload: { unit: 'degrees' } },
        { type: 'SO101_Leader', payload: { unit: 'degrees' } },
        { type: 'SO101_Follower', payload: { unit: 'radians' } },
        { type: 'other-arm', payload: { unit: 'normalized' } },
    ])('preserves degree conversion for $type with $payload', (robot) => {
        const model = createModel();

        mapJointToURDFJoint({ name: 'shoulder_pan.pos', value: 45 }, model, { 'shoulder_pan.pos': ['arm'] }, robot);

        expect(model.joints.arm.angle).toBeCloseTo(degToRad(45));
    });

    it('preserves prismatic values', () => {
        const model = createModel('prismatic');

        mapJointToURDFJoint(
            { name: 'gripper.pos', value: 0.25 },
            model,
            { 'gripper.pos': ['arm'] },
            {
                type: 'SO101_Follower',
                payload: { unit: 'normalized' },
            }
        );

        expect(model.joints.arm.jointValue[0]).toBe(0.25);
    });

    it('ignores non-position features and missing mapped joints', () => {
        const model = createModel();

        mapJointToURDFJoint({ name: 'shoulder_pan.vel', value: 45 }, model, { 'shoulder_pan.vel': ['arm'] });
        mapJointToURDFJoint({ name: 'unknown.pos', value: 45 }, model, {});
        mapJointToURDFJoint({ name: 'shoulder_pan.pos', value: 45 }, model, { 'shoulder_pan.pos': ['missing'] });

        expect(model.joints.arm.angle).toBe(0);
    });
});

describe('featureRange', () => {
    const jointMap = { 'arm.pos': ['arm'], 'gripper.pos': ['arm'] };

    it('gives revolute limits in degrees', () => {
        const range = featureRange('arm.pos', createModel(), jointMap, { type: 'trossen_wxai' });

        expect(range?.min).toBeCloseTo(-57.2958);
        expect(range?.max).toBeCloseTo(114.5916);
    });

    it('gives the normalized SO101 ranges', () => {
        const robot = { type: 'SO101_Follower', payload: { unit: 'normalized' } };

        expect(featureRange('arm.pos', createModel(), jointMap, robot)).toEqual({ min: -100, max: 100 });
        expect(featureRange('gripper.pos', createModel(), jointMap, robot)).toEqual({ min: 0, max: 100 });
    });

    it('gives prismatic limits as they are', () => {
        expect(featureRange('arm.pos', createModel('prismatic'), jointMap)).toEqual({ min: -1, max: 2 });
    });

    it('has no range for an unmapped, unlimited or non-position feature', () => {
        expect(featureRange('elbow.pos', createModel(), jointMap)).toBeUndefined();
        expect(featureRange('arm.vel', createModel(), { 'arm.vel': ['arm'] })).toBeUndefined();
        const unlimited = createModel('continuous');
        unlimited.joints.arm.limit.lower = 0;
        unlimited.joints.arm.limit.upper = 0;
        expect(featureRange('arm.pos', unlimited, jointMap)).toBeUndefined();
    });
});
