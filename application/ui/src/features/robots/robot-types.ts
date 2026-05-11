import {
    SchemaSo101RobotInput,
    SchemaSo101RobotOutput,
    SchemaTrossenSingleArmRobotInput,
    SchemaTrossenSingleArmRobotOutput,
} from '../../api/openapi-spec';

/** Union of all concrete robot output schemas (as returned by the API). */
export type SchemaRobot = SchemaSo101RobotOutput | SchemaTrossenSingleArmRobotOutput;

/** Union of all concrete robot input schemas (for create/update requests). */
export type SchemaRobotInput = SchemaSo101RobotInput | SchemaTrossenSingleArmRobotInput;

/** All possible robot type discriminators. */
export type SchemaRobotType = SchemaRobot['type'];
