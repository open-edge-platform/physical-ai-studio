import { useEffect, useState } from 'react';

import { Flex, Heading, Switch, Text, View } from '@geti-ui/ui';

import { SchemaRobotType } from '../../robot-types';
import { useRobotForm } from '../provider';
import { ConnectionField } from './components/connection-field';
import { InfoField } from './components/info-field';
import { SchemaField } from './schema-field';
import {
    asRecord,
    EMPTY_DEFINITIONS,
    EMPTY_PROPERTIES,
    fieldLabel,
    isRequiredField,
    resolveReference,
    schemaDefaults,
    updateObjectField,
} from './schema-utils';
import { FieldSchema, JsonSchema, ModelUiOptions, RobotUiItem } from './types';

const EMPTY_ITEMS: RobotUiItem[] = [];

const isUiItems = (value: unknown): value is ModelUiOptions => Array.isArray(value);

const fieldNamesOwnedByItems = (items: RobotUiItem[]): Set<string> =>
    new Set(
        items.flatMap((item) => {
            if (item.kind === 'field') {
                return [item.name];
            }
            if (item.kind === 'connection') {
                return [
                    item.bind.connection,
                    ...(item.bind.serial_number === undefined ? [] : [item.bind.serial_number]),
                ];
            }
            if (item.kind === 'section') {
                return [...fieldNamesOwnedByItems(item.items)];
            }
            return [];
        })
    );

type OnChange = (name: string, value: unknown) => void;
type IsFieldVisible = (name: string, field: FieldSchema, required: Set<string>) => boolean;
type IsRenderable = (item: RobotUiItem, properties: Record<string, FieldSchema>, required: Set<string>) => boolean;

type SchemaFormItemProps = SchemaFormItemsProps & {
    item: RobotUiItem;
};

type SchemaFormItemsProps = {
    items: RobotUiItem[];
    properties: Record<string, FieldSchema>;
    required: Set<string>;
    values: Record<string, unknown>;
    onChange: OnChange;
    robotType: SchemaRobotType;
    definitions: Record<string, FieldSchema>;
    isFieldVisible: IsFieldVisible;
    isRenderable: IsRenderable;
    renderUnownedFields: boolean;
};

type SchemaFormFieldProps = Omit<SchemaFormItemsProps, 'items' | 'renderUnownedFields'> & {
    name: string;
    field: FieldSchema;
};

const SchemaFormItem = ({ item, ...props }: SchemaFormItemProps) => {
    if (item.kind === 'info') {
        return <InfoField info={item} />;
    }
    if (item.kind === 'connection') {
        return (
            <ConnectionField
                robotType={props.robotType}
                payload={props.values}
                options={item}
                onChange={props.onChange}
            />
        );
    }
    if (item.kind === 'field') {
        const field = props.properties[item.name];
        return field === undefined ? null : <SchemaFormField {...props} name={item.name} field={field} />;
    }
    if (!props.isRenderable(item, props.properties, props.required)) {
        return null;
    }
    return (
        <Flex direction='column' gap='size-150'>
            {item.title !== undefined && <Heading level={4}>{item.title}</Heading>}
            {item.description !== undefined && <Text>{item.description}</Text>}
            <SchemaFormItems {...props} items={item.items} renderUnownedFields={false} />
        </Flex>
    );
};

const SchemaFormItems = ({ items, renderUnownedFields, ...props }: SchemaFormItemsProps) => {
    const unownedFields = renderUnownedFields
        ? (() => {
              const ownedFields = fieldNamesOwnedByItems(items);
              return Object.entries(props.properties).filter(([name]) => !ownedFields.has(name));
          })()
        : [];

    return (
        <>
            {items.map((item, index) => (
                <SchemaFormItem
                    {...props}
                    key={item.kind === 'section' ? item.id : `${item.kind}-${index}`}
                    item={item}
                    items={items}
                    renderUnownedFields={renderUnownedFields}
                />
            ))}
            {renderUnownedFields &&
                unownedFields.map(([name, field]) => (
                    <SchemaFormField {...props} key={name} name={name} field={field} />
                ))}
        </>
    );
};

const SchemaFormField = ({ name, field, ...props }: SchemaFormFieldProps) => {
    if (!props.isFieldVisible(name, field, props.required)) {
        return null;
    }

    const resolvedField = resolveReference(field, props.definitions);
    const fieldUi = resolvedField['x-physicalai-ui'];
    const isRequired = isRequiredField(name, resolvedField, props.required);
    if (resolvedField.properties !== undefined) {
        const nestedItems = isUiItems(fieldUi) ? fieldUi : EMPTY_ITEMS;
        return (
            <View backgroundColor='gray-50' borderColor='gray-200' borderWidth='thin' padding='size-150'>
                <Flex direction='column' gap='size-150'>
                    <Heading level={4}>{fieldLabel(name, field)}</Heading>
                    <SchemaFormItems
                        {...props}
                        items={nestedItems}
                        properties={resolvedField.properties}
                        required={new Set(resolvedField.required ?? [])}
                        values={asRecord(props.values[name])}
                        onChange={(nestedName, nestedValue) =>
                            props.onChange(name, updateObjectField(props.values[name], nestedName, nestedValue))
                        }
                        renderUnownedFields
                    />
                </Flex>
            </View>
        );
    }

    if (resolvedField.type === 'object') {
        return null;
    }

    return (
        <SchemaField
            name={name}
            schema={resolvedField}
            value={props.values[name]}
            isRequired={isRequired}
            onChange={(value) => props.onChange(name, value)}
        />
    );
};

export const SchemaForm = ({ schema }: { schema: JsonSchema }) => {
    const { activeType, payload, setPayload, updatePayloadField } = useRobotForm();
    const [showAdvanced, setShowAdvanced] = useState(false);
    const properties = schema.properties ?? EMPTY_PROPERTIES;
    const definitions = schema.$defs ?? EMPTY_DEFINITIONS;
    const required = new Set(schema.required ?? []);
    const items = isUiItems(schema['x-physicalai-ui']) ? schema['x-physicalai-ui'] : EMPTY_ITEMS;

    useEffect(() => {
        if (Object.keys(payload).length !== 0) {
            return;
        }
        const defaults = schemaDefaults(properties, definitions);
        if (Object.keys(defaults).length !== 0) {
            setPayload(defaults);
        }
    }, [definitions, payload, properties, setPayload]);

    const isFieldVisible: IsFieldVisible = (name, field, fieldRequired) => {
        const resolvedField = resolveReference(field, definitions);
        const fieldUi = resolvedField['x-physicalai-ui'];

        const isRequired = isRequiredField(name, resolvedField, fieldRequired);
        if (!isRequired && !isUiItems(fieldUi) && fieldUi?.advanced_configuration === true && !showAdvanced) {
            return false;
        }

        return resolvedField.type !== 'object' || resolvedField.properties !== undefined;
    };

    const isRenderable: IsRenderable = (item, itemProperties, itemRequired) => {
        if (item.kind === 'info' || item.kind === 'connection') {
            return true;
        }
        if (item.kind === 'field') {
            const field = itemProperties[item.name];
            return field !== undefined && isFieldVisible(item.name, field, itemRequired);
        }
        return item.items.some((child) => isRenderable(child, itemProperties, itemRequired));
    };

    return (
        <Flex direction='column' gap='size-200'>
            <Flex justifyContent='end'>
                <Switch isSelected={showAdvanced} onChange={setShowAdvanced} isHidden>
                    Show advanced options
                </Switch>
            </Flex>
            <SchemaFormItems
                items={items}
                properties={properties}
                required={required}
                values={payload}
                onChange={updatePayloadField}
                robotType={activeType!}
                definitions={definitions}
                isFieldVisible={isFieldVisible}
                isRenderable={isRenderable}
                renderUnownedFields
            />
        </Flex>
    );
};
