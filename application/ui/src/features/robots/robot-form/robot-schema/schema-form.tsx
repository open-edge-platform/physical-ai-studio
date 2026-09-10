import { ReactNode, useEffect } from 'react';

import { Disclosure, DisclosurePanel, DisclosureTitle, Flex, Heading, Text, View } from '@geti-ui/ui';
import { partition } from 'lodash-es';

import { SchemaRobotType } from '../../robot-types';
import { useRobotForm } from '../provider';
import { CalibrationField } from './components/calibration-field';
import { ConnectionField } from './components/connection-field';
import { InfoField } from './components/info-field';
import { IpAddressField } from './components/ip-address-field';
import { SchemaField } from './schema-field';
import {
    asRecord,
    EMPTY_DEFINITIONS,
    EMPTY_PROPERTIES,
    fieldContextualInfo,
    fieldLabel,
    isRequiredField,
    resolveReference,
    schemaDefaults,
    updateObjectField,
} from './schema-utils';
import { ContextualInfo, FieldSchema, JsonSchema, ModelUiOptions, RobotUiItem } from './types';

import classes from './schema-form.module.css';

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
            if (item.kind === 'ip_address') {
                return [item.name];
            }
            if (item.kind === 'calibration') {
                return [item.name];
            }
            if (item.kind === 'section') {
                return [...fieldNamesOwnedByItems(item.items)];
            }
            return [];
        })
    );

type OnChange = (name: string, value: unknown) => void;
type IsFieldVisible = (name: string, field: FieldSchema, required: Set<string>, showAdvanced: boolean) => boolean;
type IsFieldEnabled = (name: string, field: FieldSchema, required: Set<string>, showAdvanced: boolean) => boolean;
type IsRenderable = (
    item: RobotUiItem,
    properties: Record<string, FieldSchema>,
    required: Set<string>,
    showAdvanced: boolean
) => boolean;

type ItemEntry = {
    item: RobotUiItem;
    index: number;
};

type SchemaFormItemProps = SchemaFormItemsProps & {
    item: RobotUiItem;
    showAdvanced: boolean;
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
    isFieldEnabled: IsFieldEnabled;
    isRenderable: IsRenderable;
    renderUnownedFields: boolean;
};

type SchemaFormFieldProps = Omit<SchemaFormItemsProps, 'items' | 'renderUnownedFields'> & {
    name: string;
    field: FieldSchema;
    info?: ContextualInfo;
    showAdvanced: boolean;
};

type SchemaFormItemListProps = SchemaFormItemsProps & {
    entries: ItemEntry[];
    showAdvanced: boolean;
};

type SchemaFormUnownedFieldListProps = Omit<SchemaFormItemsProps, 'items' | 'renderUnownedFields'> & {
    fields: [string, FieldSchema][];
    showAdvanced: boolean;
};

const getResolvedField = ({ properties, definitions }: SchemaFormItemsProps, name: string) => {
    const field = properties[name];
    return field === undefined ? undefined : resolveReference(field, definitions);
};

const asFieldSchema = (value: FieldSchema | boolean | undefined): FieldSchema | undefined =>
    typeof value === 'object' && value !== null && !Array.isArray(value) ? value : undefined;

const AdvancedOptions = ({ children }: { children: ReactNode }) => (
    <Disclosure isQuiet UNSAFE_className={classes.advancedDisclosure}>
        <DisclosureTitle UNSAFE_className={classes.advancedDisclosureTitle}>
            <Text UNSAFE_className={classes.advancedDisclosureText}>Advanced options</Text>
        </DisclosureTitle>
        <DisclosurePanel
            UNSAFE_className={classes.advancedDisclosurePanel}

            UNSAFE_style={{ paddingBlock: 0 }}
        >
            {children}
        </DisclosurePanel>
    </Disclosure>
);

const SchemaFormItemList = ({ entries, showAdvanced, ...props }: SchemaFormItemListProps) => (
    <>
        {entries.map(({ item, index }) => (
            <SchemaFormItem
                {...props}
                key={item.kind === 'section' ? item.id : `${item.kind}-${index}`}
                item={item}
                showAdvanced={showAdvanced}
            />
        ))}
    </>
);

const SchemaFormUnownedFieldList = ({ fields, showAdvanced, ...props }: SchemaFormUnownedFieldListProps) => (
    <>
        {fields.map(([name, field]) => (
            <SchemaFormField {...props} key={name} name={name} field={field} showAdvanced={showAdvanced} />
        ))}
    </>
);

const SchemaFormItem = ({ item, ...props }: SchemaFormItemProps) => {
    if (item.kind === 'info') {
        return <InfoField info={item} />;
    }
    if (item.kind === 'connection') {
        const field = getResolvedField(props, item.bind.connection);
        if (field === undefined) {
            return null;
        }
        return (
            <ConnectionField
                robotType={props.robotType}
                payload={props.values}
                options={{ ...item, info: item.info ?? fieldContextualInfo(field) }}
                isRequired={isRequiredField(item.bind.connection, field, props.required)}
                onChange={props.onChange}
            />
        );
    }
    if (item.kind === 'ip_address') {
        const field = getResolvedField(props, item.name);
        if (field === undefined) {
            return null;
        }
        return (
            <IpAddressField
                robotType={props.robotType}
                payload={props.values}
                options={{ ...item, info: item.info ?? fieldContextualInfo(field) }}
                isRequired={isRequiredField(item.name, field, props.required)}
                onChange={props.onChange}
            />
        );
    }
    if (item.kind === 'calibration') {
        const field = getResolvedField(props, item.name);
        if (field === undefined) {
            return null;
        }
        if (!props.isFieldEnabled(item.name, field, props.required, props.showAdvanced)) {
            return null;
        }

        return (
            <CalibrationField
                label={item.label ?? fieldLabel(item.name, field)}
                description={item.description ?? field.description}
                info={item.info ?? fieldContextualInfo(field)}
                isRequired={isRequiredField(item.name, field, props.required)}
                value={props.values[item.name]}
                valueSchema={asFieldSchema(field.additionalProperties)}
                definitions={props.definitions}
                onChange={(value) => props.onChange(item.name, value)}
            />
        );
    }
    if (item.kind === 'field') {
        const field = props.properties[item.name];
        return field === undefined ? null : (
            <SchemaFormField
                {...props}
                name={item.name}
                field={field}
                info={item.info}
                showAdvanced={props.showAdvanced}
            />
        );
    }
    if (!props.isRenderable(item, props.properties, props.required, true)) {
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
    const indexedItems = items.map((item, index) => ({ item, index }));
    const renderableItems = indexedItems.filter(({ item }) =>
        props.isRenderable(item, props.properties, props.required, true)
    );

    const [basicItems, advancedItems] = partition(
        renderableItems,
        ({ item }) => item.kind === 'section' || props.isRenderable(item, props.properties, props.required, false)
    );
    const [basicUnownedFields, advancedUnownedFields] = renderUnownedFields
        ? partition(
              unownedFields.filter(([name, field]) => props.isFieldVisible(name, field, props.required, true)),
              ([name, field]) => props.isFieldVisible(name, field, props.required, false)
          )
        : [[], []];

    const hasAdvancedFields = advancedItems.length !== 0 || advancedUnownedFields.length !== 0;

    return (
        <Flex direction='column' gap='size-100'>
            <SchemaFormItemList
                {...props}
                entries={basicItems}
                items={items}
                renderUnownedFields={renderUnownedFields}
                showAdvanced={false}
            />
            <SchemaFormUnownedFieldList {...props} fields={basicUnownedFields} showAdvanced={false} />
            {hasAdvancedFields && (
                <AdvancedOptions>
                    <Flex direction='column' gap='size-100'>
                        <SchemaFormItemList
                            {...props}
                            entries={advancedItems}
                            items={items}
                            renderUnownedFields={renderUnownedFields}
                            showAdvanced
                        />
                        <SchemaFormUnownedFieldList {...props} fields={advancedUnownedFields} showAdvanced />
                    </Flex>
                </AdvancedOptions>
            )}
        </Flex>
    );
};

const SchemaFormField = ({ name, field, ...props }: SchemaFormFieldProps) => {
    if (!props.isFieldVisible(name, field, props.required, props.showAdvanced)) {
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
            info={props.info ?? fieldContextualInfo(resolvedField)}
            value={props.values[name]}
            isRequired={isRequired}
            onChange={(value) => props.onChange(name, value)}
        />
    );
};

export const SchemaForm = ({ schema }: { schema: JsonSchema }) => {
    const { activeType, payload, setPayload, updatePayloadField } = useRobotForm();
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

    const isFieldVisible: IsFieldVisible = (name, field, fieldRequired, showAdvanced) => {
        const resolvedField = resolveReference(field, definitions);
        if (!isFieldEnabled(name, resolvedField, fieldRequired, showAdvanced)) {
            return false;
        }

        return resolvedField.type !== 'object' || resolvedField.properties !== undefined;
    };

    const isFieldEnabled: IsFieldEnabled = (name, field, fieldRequired, showAdvanced) => {
        const resolvedField = resolveReference(field, definitions);
        const fieldUi = resolvedField['x-physicalai-ui'];
        const isRequired = isRequiredField(name, resolvedField, fieldRequired);

        return isRequired || isUiItems(fieldUi) || fieldUi?.advanced_configuration !== true || showAdvanced;
    };

    const isRenderable: IsRenderable = (item, itemProperties, itemRequired, showAdvanced) => {
        if (item.kind === 'info' || item.kind === 'connection' || item.kind === 'ip_address') {
            return true;
        }
        if (item.kind === 'calibration') {
            const field = itemProperties[item.name];
            return field !== undefined && isFieldEnabled(item.name, field, itemRequired, showAdvanced);
        }
        if (item.kind === 'field') {
            const field = itemProperties[item.name];
            return field !== undefined && isFieldVisible(item.name, field, itemRequired, showAdvanced);
        }
        return item.items.some((child) => isRenderable(child, itemProperties, itemRequired, showAdvanced));
    };

    return (
        <SchemaFormItems
            items={items}
            properties={properties}
            required={required}
            values={payload}
            onChange={updatePayloadField}
            robotType={activeType!}
            definitions={definitions}
            isFieldVisible={isFieldVisible}
            isFieldEnabled={isFieldEnabled}
            isRenderable={isRenderable}
            renderUnownedFields
        />
    );
};
