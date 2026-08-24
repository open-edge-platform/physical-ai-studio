import type { operations, SchemaUnavailableRobot } from '../../api/openapi-spec';

type ListProjectRobotsOperation = operations['list_project_robots_api_projects__project_id__robots_get'];
type CreateProjectRobotOperation = operations['create_project_robot_api_projects__project_id__robots_post'];

/** Union of all concrete robot output schemas (as returned by list/get/create robot APIs). */
export type SchemaRobot = ListProjectRobotsOperation['responses'][200]['content']['application/json'][number];

/** Union of all concrete robot input schemas (for create robot requests). */
export type SchemaRobotInput = CreateProjectRobotOperation['requestBody']['content']['application/json'];

/** Single robot response payload from create robot API. */
export type SchemaRobotCreateResponse = CreateProjectRobotOperation['responses'][201]['content']['application/json'];

/** All possible robot type discriminators. */
export type AvailableSchemaRobot = Exclude<SchemaRobot, SchemaUnavailableRobot>;

/** All robot type discriminators, including persisted types from unavailable plugins. */
export type SchemaRobotType = SchemaRobot['type'];

/** Robot type discriminators that are currently installed and can be configured. */
export type ConfigurableRobotType = AvailableSchemaRobot['type'];

export const isUnavailableRobot = (robot: SchemaRobot): robot is SchemaUnavailableRobot =>
    'unavailable' in robot && robot.unavailable;
