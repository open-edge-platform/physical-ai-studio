import { FieldSchema } from './types';

export const EMPTY_PROPERTIES: Record<string, FieldSchema> = {};
export const EMPTY_DEFINITIONS: Record<string, FieldSchema> = {};

export const fieldLabel = (name: string, schema: FieldSchema) =>
    schema.title ?? name.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());

export const asRecord = (value: unknown): Record<string, unknown> =>
    typeof value === 'object' && value !== null && !Array.isArray(value) ? (value as Record<string, unknown>) : {};

export const resolveReference = (schema: FieldSchema, definitions: Record<string, FieldSchema>): FieldSchema => {
    if (schema.$ref === undefined) {
        return schema;
    }

    const definitionName = schema.$ref.replace('#/$defs/', '');
    const definition = definitions[definitionName];
    return definition === undefined ? schema : { ...definition, ...schema };
};

export const isRequiredField = (name: string, schema: FieldSchema, required: Set<string>) => {
    const uiOptions = schema['x-physicalai-ui'];
    return required.has(name) || (!Array.isArray(uiOptions) && uiOptions?.required === true);
};

const schemaDefault = (schema: FieldSchema, definitions: Record<string, FieldSchema>): unknown => {
    const resolvedSchema = resolveReference(schema, definitions);
    if (resolvedSchema.default !== undefined) {
        return resolvedSchema.default;
    }
    if (resolvedSchema.properties === undefined) {
        return undefined;
    }

    const defaults = schemaDefaults(resolvedSchema.properties, definitions);
    return Object.keys(defaults).length === 0 ? undefined : defaults;
};

export const schemaDefaults = (
    properties: Record<string, FieldSchema>,
    definitions: Record<string, FieldSchema>
): Record<string, unknown> =>
    Object.fromEntries(
        Object.entries(properties).flatMap(([name, field]) => {
            const defaultValue = schemaDefault(field, definitions);
            return defaultValue === undefined ? [] : [[name, defaultValue]];
        })
    );

export const updateObjectField = (value: unknown, name: string, fieldValue: unknown): Record<string, unknown> => ({
    ...asRecord(value),
    [name]: fieldValue,
});
