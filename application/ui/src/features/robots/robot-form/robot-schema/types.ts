export type InfoItem = {
    kind: 'info';
    title?: string;
    text: string;
    variant?: 'info' | 'warning';
};

export type RobotUiConnectionBinding = {
    connection: string;
    serial_number?: string;
};

export type ConnectionItem = {
    kind: 'connection';
    label?: string;
    description?: string;
    device_discovery?: boolean;
    identify?: boolean;
    manual_entry?: boolean;
    bind: RobotUiConnectionBinding;
};

export type FieldItem = {
    kind: 'field';
    name: string;
};

export type SectionItem = {
    kind: 'section';
    id: string;
    title?: string;
    description?: string;
    items: RobotUiItem[];
};

export type RobotUiItem = InfoItem | ConnectionItem | FieldItem | SectionItem;

export type FieldOptions = {
    required?: boolean;
    advanced_configuration?: boolean;
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
