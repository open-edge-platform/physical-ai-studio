import { SchemaRobotType } from '../../robot-types';

export type InfoItem = {
    kind: 'info';
    title?: string;
    text: string;
    variant?: 'info' | 'warning';
};

export type ContextualInfo = {
    title?: string;
    description: string;
    link_url?: string;
    variant?: 'info' | 'help';
};

export type RobotUiConnectionBinding = {
    connection: string;
    serial_number?: string;
};

export type ConnectionItem = {
    kind: 'connection';
    label?: string;
    description?: string;
    info?: ContextualInfo;
    device_discovery?: boolean;
    identify?: boolean;
    manual_entry?: boolean;
    bind: RobotUiConnectionBinding;
};

export type IpAddressItem = {
    kind: 'ip_address';
    name: string;
    label?: string;
    description?: string;
    info?: ContextualInfo;
    identify?: boolean;
    identify_robot_type?: SchemaRobotType;
};

export type CalibrationItem = {
    kind: 'calibration';
    name: string;
    label?: string;
    description?: string;
    info?: ContextualInfo;
};

export type FieldItem = {
    kind: 'field';
    name: string;
    info?: ContextualInfo;
};

export type SectionItem = {
    kind: 'section';
    id: string;
    title?: string;
    description?: string;
    items: RobotUiItem[];
};

export type RobotUiItem = InfoItem | ConnectionItem | IpAddressItem | CalibrationItem | FieldItem | SectionItem;

export type FieldOptions = {
    required?: boolean;
    advanced_configuration?: boolean;
    info?: ContextualInfo;
};

export type ModelUiOptions = RobotUiItem[];

export type FieldSchema = {
    type?: string;
    title?: string;
    description?: string;
    default?: unknown;
    enum?: unknown[];
    $ref?: string;
    properties?: Record<string, FieldSchema>;
    additionalProperties?: FieldSchema | boolean;
    required?: string[];
    ['x-physicalai-ui']?: FieldOptions | ModelUiOptions;
};

export type JsonSchema = {
    type?: string;
    properties?: Record<string, FieldSchema>;
    required?: string[];
    $defs?: Record<string, FieldSchema>;
    ['x-physicalai-ui']?: ModelUiOptions;
};
