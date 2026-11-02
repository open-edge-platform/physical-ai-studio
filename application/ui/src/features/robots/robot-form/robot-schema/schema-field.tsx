import { Flex, Item, Picker, Switch, Text, TextField } from '@geti-ui/ui';

import { FieldContextualHelp } from './components/field-contextual-help';
import { fieldContextualInfo, fieldLabel } from './schema-utils';
import { ContextualInfo, FieldSchema } from './types';

type FieldProps = {
    name: string;
    schema: FieldSchema;
    value: unknown;
    isRequired: boolean;
    info?: ContextualInfo;
    onChange: (value: unknown) => void;
};

const commonProps = ({
    name,
    schema,
    isRequired,
    info,
}: Pick<FieldProps, 'name' | 'schema' | 'isRequired' | 'info'>) => {
    const contextualInfo = info ?? fieldContextualInfo(schema);

    return {
        label: fieldLabel(name, schema),
        description: schema.description,
        contextualHelp: contextualInfo === undefined ? undefined : <FieldContextualHelp info={contextualInfo} />,
        isRequired,
        width: '100%' as const,
    };
};

const EnumPickerField = ({ name, schema, value, isRequired, onChange, info }: FieldProps) => (
    <Picker
        {...commonProps({ name, schema, isRequired, info })}
        selectedKey={String(value ?? '')}
        onSelectionChange={onChange}
    >
        {(schema.enum ?? []).map((option) => (
            <Item key={String(option)}>{String(option)}</Item>
        ))}
    </Picker>
);

const BooleanField = ({ name, schema, value, isRequired, onChange, info }: FieldProps) => {
    const contextualInfo = info ?? fieldContextualInfo(schema);

    return (
        <Flex direction='column' gap='size-50'>
            <Flex alignItems='center' gap='size-75'>
                <Switch isRequired={isRequired} isSelected={Boolean(value)} onChange={onChange}>
                    {fieldLabel(name, schema)}
                </Switch>
                {contextualInfo === undefined ? null : <FieldContextualHelp info={contextualInfo} />}
            </Flex>
            {schema.description !== undefined && schema.description !== '' && <Text>{schema.description}</Text>}
        </Flex>
    );
};

const TextFieldValue = ({ name, schema, value, isRequired, onChange, info }: FieldProps) => {
    const isNumeric = schema.type === 'integer' || schema.type === 'number';

    const parseValue = (next: string): unknown => {
        if (!isNumeric) {
            return next;
        }

        const parsed = schema.type === 'integer' ? Number.parseInt(next, 10) : Number.parseFloat(next);
        return Number.isNaN(parsed) ? undefined : parsed;
    };

    return (
        <TextField
            {...commonProps({ name, schema, isRequired, info })}
            type={isNumeric ? 'number' : 'text'}
            value={value === undefined || value === null ? '' : String(value)}
            onChange={(next) => onChange(parseValue(next))}
        />
    );
};

export const SchemaField = (props: FieldProps) => {
    if (props.schema.enum) {
        return <EnumPickerField {...props} />;
    }
    if (props.schema.type === 'boolean') {
        return <BooleanField {...props} />;
    }
    return <TextFieldValue {...props} />;
};
