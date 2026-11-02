/**
 * Camera slots a policy expects, in the order the policy reads them.
 *
 * Some policies are pretrained on a fixed camera order: the checkpoint's first
 * camera slot always sees the same kind of view. A dataset has no such notion —
 * its cameras are whatever the environment happened to have — so the slots below
 * are the fixed side of the mapping the user fills in on the feature-mapping step.
 *
 * The order of each list is the slot order: index 0 is the policy's first camera.
 * A policy without an entry here consumes the dataset's cameras as they come and
 * has nothing to map.
 */
export interface PolicyCameraSlot {
    /** Stable id; also the camera key the pretrained checkpoint declares. */
    id: string;
    name: string;
    /** Shown in the slot's contextual help. */
    description: string;
    /** A slot that must be filled for the policy to train sensibly. */
    isRequired: boolean;
}

const SMOLVLA_CAMERA_SLOTS: PolicyCameraSlot[] = [
    {
        id: 'camera1',
        name: 'Overview',
        description:
            'The scene view: a fixed camera that sees the workspace and the robot in it. ' +
            'SmolVLA is pretrained with this view in its first camera slot, so mapping a wrist ' +
            'camera here means the policy sees something quite different from what it was trained on.',
        isRequired: true,
    },
    {
        id: 'camera2',
        name: 'Gripper',
        description:
            'The wrist or gripper view: a camera mounted on the arm that sees what the gripper is ' +
            'about to grasp. Leave it empty when the robot has no wrist camera — the slot is then ' +
            'filled with a masked empty image.',
        isRequired: false,
    },
    {
        id: 'camera3',
        name: 'Extra',
        description:
            'A third view — a second scene angle or a second wrist camera. Leave it empty when the ' +
            'setup has only two cameras.',
        isRequired: false,
    },
];

export const POLICY_CAMERA_SLOTS: Record<string, PolicyCameraSlot[]> = {
    smolvla: SMOLVLA_CAMERA_SLOTS,
};

export const getPolicyCameraSlots = (policy: string): PolicyCameraSlot[] => POLICY_CAMERA_SLOTS[policy] ?? [];

/** Whether a policy reads its cameras in a fixed order the dataset has to be mapped onto. */
export const requiresFeatureMapping = (policy: string): boolean => getPolicyCameraSlots(policy).length > 0;
