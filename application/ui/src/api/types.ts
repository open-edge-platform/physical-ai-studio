import {
    SchemaBaslerCameraInput,
    SchemaGenicamCameraInput,
    SchemaIpCameraInput,
    SchemaRealsenseCameraInput,
    SchemaSimulationCameraInput,
    SchemaUsbCameraInput,
} from './openapi-spec';

//
export type SchemaProjectCamera =
    | SchemaUsbCameraInput
    | SchemaBaslerCameraInput
    | SchemaGenicamCameraInput
    | SchemaIpCameraInput
    | SchemaRealsenseCameraInput
    | SchemaSimulationCameraInput
    | SchemaUsbCameraInput;
