import { useEffect, useMemo, useRef, useState } from 'react';

import { degToRad } from 'three/src/math/MathUtils.js';

import { useRobotModels } from '../robot-models-context';

// ---------------------------------------------------------------------------
// Shared easing function
// ---------------------------------------------------------------------------

/** Quadratic ease-in-out: smooth acceleration then deceleration. */
const easeInOut = (t: number): number => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);

// ---------------------------------------------------------------------------
// Centering animation
// ---------------------------------------------------------------------------

/**
 * Rest/home pose of the SO101 arm (degrees).
 * This is the position the robot would be in when first powered on
 * with no calibration — a natural folded posture.
 */
const HOME_POSITION_STATE: Record<string, number> = {
    shoulder_pan: 0,
    shoulder_lift: -90,
    elbow_flex: 90,
    wrist_flex: 60,
    wrist_roll: 50,
    gripper: 85,
};

/**
 * Centered (zero) pose — the position the user should move the robot to
 * during the homing/centering step.
 */
const TARGET_POSITION_STATE: Record<string, number> = {
    shoulder_pan: 0,
    shoulder_lift: 0,
    elbow_flex: 0,
    wrist_flex: 0,
    wrist_roll: 0,
    gripper: -10,
};

const CENTERING_DURATION_MS = 4000;

/**
 * Animates the 3D URDF model from a rest/home pose to the centered (zero)
 * position over 4 seconds using ease-in-out interpolation.
 *
 * Adapted from the experimental centering-step.tsx (commit c3f677b0).
 */
export const useCenteringAnimation = (enabled: boolean) => {
    const { models } = useRobotModels();
    const animationRef = useRef<number | null>(null);

    useEffect(() => {
        if (!enabled || models.length === 0) return;

        const startTime = performance.now();

        const animate = (currentTime: number) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / CENTERING_DURATION_MS, 1);
            const eased = easeInOut(progress);

            models.forEach((model) => {
                Object.values(model.joints).forEach((joint) => {
                    const homeValue = HOME_POSITION_STATE[joint.urdfName] ?? 0;
                    const targetValue = TARGET_POSITION_STATE[joint.urdfName] ?? 0;
                    const interpolated = homeValue + (targetValue - homeValue) * eased;
                    joint.setJointValue(degToRad(interpolated));
                });
            });

            if (progress < 1) {
                animationRef.current = requestAnimationFrame(animate);
            }
        };

        animationRef.current = requestAnimationFrame(animate);

        return () => {
            if (animationRef.current !== null) {
                cancelAnimationFrame(animationRef.current);
                animationRef.current = null;
            }
        };
    }, [enabled, models]);
};

// ---------------------------------------------------------------------------
// Range-of-motion animation
// ---------------------------------------------------------------------------

const RANGE_TRANSITION_MS = 2000;
const RANGE_PAUSE_MS = 500;

/**
 * Continuously cycles the 3D URDF model through each joint's full range
 * of motion: home -> upper limit -> lower limit, for every movable joint.
 *
 * Adapted from the experimental MovementPreview (commit c3f677b0).
 */
export const useRangeOfMotionAnimation = (enabled: boolean) => {
    const { models } = useRobotModels();
    const animationRef = useRef<number | null>(null);
    const cycleTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const [currentStep, setCurrentStep] = useState(0);

    const model = models.at(0);
    const joints = model?.joints ?? {};

    const changeableJoints = useMemo(
        () =>
            Object.keys(joints).filter((jointName) => {
                const joint = joints[jointName];
                return joint.jointType !== 'fixed';
            }),
        [joints]
    );

    // Build the animation sequence: for each movable joint, go home -> upper -> lower
    const animationSequence = useMemo(() => {
        const sequence: Array<{ name: string; position: Record<string, number> }> = [];

        changeableJoints.forEach((jointName) => {
            const joint = joints[jointName];

            // Return to home (all zeros)
            sequence.push({
                name: 'Home',
                position: Object.fromEntries(Object.keys(joints).map((j) => [j, 0])),
            });

            // Upper limit
            sequence.push({
                name: `${jointName} Upper`,
                position: { [jointName]: joint.limit.upper },
            });

            // Lower limit
            sequence.push({
                name: `${jointName} Lower`,
                position: { [jointName]: joint.limit.lower },
            });
        });

        return sequence;
    }, [joints, changeableJoints]);

    // Reset step counter when animation is re-enabled
    useEffect(() => {
        if (enabled) {
            setCurrentStep(0);
        }
    }, [enabled]);

    // Animate to the current step's target position
    useEffect(() => {
        if (!enabled || models.length === 0 || animationSequence.length === 0) return;

        const targetPosition = animationSequence[currentStep].position;

        // Capture starting positions for interpolation
        const startPositions: Record<number, Record<string, number>> = {};
        models.forEach((m, index) => {
            startPositions[index] = {};
            Object.values(m.joints).forEach((joint) => {
                startPositions[index][joint.urdfName] = joint.angle;
            });
        });

        const startTime = performance.now();

        const animate = (currentTime: number) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / RANGE_TRANSITION_MS, 1);
            const eased = easeInOut(progress);

            models.forEach((m, index) => {
                Object.values(m.joints).forEach((joint) => {
                    const startValue = startPositions[index][joint.urdfName] ?? 0;
                    const targetValue = targetPosition[joint.urdfName];

                    if (targetValue !== undefined) {
                        const interpolated = startValue + (targetValue - startValue) * eased;
                        joint.setJointValue(interpolated);
                    }
                });
            });

            if (progress < 1) {
                animationRef.current = requestAnimationFrame(animate);
            } else {
                // Done — pause then advance to next step
                cycleTimeoutRef.current = setTimeout(() => {
                    setCurrentStep((prev) => (prev + 1) % animationSequence.length);
                }, RANGE_PAUSE_MS);
            }
        };

        animationRef.current = requestAnimationFrame(animate);

        return () => {
            if (animationRef.current !== null) {
                cancelAnimationFrame(animationRef.current);
                animationRef.current = null;
            }
            if (cycleTimeoutRef.current !== null) {
                clearTimeout(cycleTimeoutRef.current);
                cycleTimeoutRef.current = null;
            }
        };
    }, [enabled, models, animationSequence, currentStep]);
};
