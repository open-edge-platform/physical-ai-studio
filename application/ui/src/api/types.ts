import {
    SchemaBaslerCameraInput,
    SchemaGenicamCameraInput,
    SchemaIpCameraInput,
    SchemaRealsenseCameraInput,
    SchemaUsbCameraInput,
} from './openapi-spec';

//
export type SchemaProjectCamera =
    | SchemaUsbCameraInput
    | SchemaBaslerCameraInput
    | SchemaGenicamCameraInput
    | SchemaIpCameraInput
    | SchemaRealsenseCameraInput
    | SchemaUsbCameraInput;
