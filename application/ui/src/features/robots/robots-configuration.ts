import { SchemaRobot, SchemaRobotType } from './robot-types';

export const isFollower = (robot: SchemaRobot) => {
    return (
        robot.type === 'SO101_Follower' ||
        robot.type === 'Trossen_WidowXAI_Follower' ||
        robot.type === 'Trossen_Bimanual_WidowXAI_Follower'
    );
};

export const isLeader = (robot: SchemaRobot) => {
    return (
        robot.type === 'SO101_Leader' ||
        robot.type === 'Trossen_WidowXAI_Leader' ||
        robot.type === 'Trossen_Bimanual_WidowXAI_Leader'
    );
};

/** Resolve a `SchemaRobotType` to its URDF asset path. */
export const urdfPathForType = (robotType: SchemaRobotType): string => {
    if (robotType === 'Trossen_Bimanual_WidowXAI_Follower' || robotType === 'Trossen_Bimanual_WidowXAI_Leader') {
        return '/widowx/urdf/generated/stationary_ai.urdf';
    }

    if (robotType !== undefined && robotType.toLowerCase().includes('trossen')) {
        return '/widowx/urdf/generated/wxai/wxai_follower.urdf';
    }
    return '/SO101/so101_new_calib.urdf';
};

const SO101_TO_URDF = {
    'shoulder_pan.pos': ['shoulder_pan'],
    'shoulder_lift.pos': ['shoulder_lift'],
    'elbow_flex.pos': ['elbow_flex'],
    'wrist_flex.pos': ['wrist_flex'],
    'wrist_roll.pos': ['wrist_roll'],
    'gripper.pos': ['gripper'],
};
const TROSSEN_TO_URDF = {
    'shoulder_pan.pos': ['joint_0'],
    'shoulder_lift.pos': ['joint_1'],
    'elbow_flex.pos': ['joint_2'],
    'wrist_flex.pos': ['joint_3'],
    'wrist_yaw.pos': ['joint_4'],
    'wrist_roll.pos': ['joint_5'],
    'gripper.pos': ['left_carriage_joint', 'right_carriage_joint'],
};

const BIMANUAL_TROSSEN_TO_URDF = {
    'left_shoulder_pan.pos': ['follower_left_joint_0'],
    'left_shoulder_lift.pos': ['follower_left_joint_1'],
    'left_elbow_flex.pos': ['follower_left_joint_2'],
    'left_wrist_flex.pos': ['follower_left_joint_3'],
    'left_wrist_yaw.pos': ['follower_left_joint_4'],
    'left_wrist_roll.pos': ['follower_left_joint_5'],
    'left_gripper.pos': ['follower_left_left_carriage_joint', 'follower_left_right_carriage_joint'],

    'right_shoulder_pan.pos': ['follower_right_joint_0'],
    'right_shoulder_lift.pos': ['follower_right_joint_1'],
    'right_elbow_flex.pos': ['follower_right_joint_2'],
    'right_wrist_flex.pos': ['follower_right_joint_3'],
    'right_wrist_yaw.pos': ['follower_right_joint_4'],
    'right_wrist_roll.pos': ['follower_right_joint_5'],
    'right_gripper.pos': ['follower_right_left_carriage_joint', 'follower_right_right_carriage_joint'],
};

export const ROBOT_TYPE_TO_URDF_MAP: Record<SchemaRobotType, Record<string, string[]>> = {
    SO101_Follower: SO101_TO_URDF,
    SO101_Leader: SO101_TO_URDF,
    Trossen_WidowXAI_Leader: TROSSEN_TO_URDF,
    Trossen_WidowXAI_Follower: TROSSEN_TO_URDF,
    Trossen_Bimanual_WidowXAI_Leader: BIMANUAL_TROSSEN_TO_URDF,
    Trossen_Bimanual_WidowXAI_Follower: BIMANUAL_TROSSEN_TO_URDF,
};
