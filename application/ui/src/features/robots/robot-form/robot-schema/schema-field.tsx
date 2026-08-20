import { Flex, Item, Picker, Switch, Text, TextField } from '@geti-ui/ui';

import { fieldLabel } from './schema-utils';
import { FieldSchema } from './types';

type FieldProps = {
    name: string;
    schema: FieldSchema;
    value: unknown;
    isRequired: boolean;
    onChange: (value: unknown) => void;
};

const commonProps = ({ name, schema, isRequired }: Pick<FieldProps, 'name' | 'schema' | 'isRequired'>) => ({
    label: fieldLabel(name, schema),
    description: schema.description,
    isRequired,
    width: '100%' as const,
});

const EnumPickerField = ({ name, schema, value, isRequired, onChange }: FieldProps) => (
    <Picker
        {...commonProps({ name, schema, isRequired })}
        selectedKey={String(value ?? '')}
        onSelectionChange={onChange}
    >
        {(schema.enum ?? []).map((option) => (
            <Item key={String(option)}>{String(option)}</Item>
        ))}
    </Picker>
);

const BooleanField = ({ name, schema, value, isRequired, onChange }: FieldProps) => (
    <Flex direction='column' gap='size-50'>
        <Switch isRequired={isRequired} isSelected={Boolean(value)} onChange={onChange}>
            {fieldLabel(name, schema)}
        </Switch>
        {schema.description !== undefined && schema.description !== '' && <Text>{schema.description}</Text>}
    </Flex>
);

const TextFieldValue = ({ name, schema, value, isRequired, onChange }: FieldProps) => {
    const isNumeric = schema.type === 'integer' || schema.type === 'number';
    return (
        <TextField
            {...commonProps({ name, schema, isRequired })}
            type={isNumeric ? 'number' : 'text'}
            value={value === undefined || value === null ? '' : String(value)}
            onChange={(next) =>
                onChange(
                    schema.type === 'integer'
                        ? Number.parseInt(next, 10)
                        : schema.type === 'number'
                          ? Number.parseFloat(next)
                          : next
                )
            }
        />
    );
};

export const SchemaField = (props: FieldProps) => {
    if (props.schema.enum) return <EnumPickerField {...props} />;
    if (props.schema.type === 'boolean') return <BooleanField {...props} />;
    return <TextFieldValue {...props} />;
};
