import { TextField } from '@geti/ui';

interface NameFieldProps {
    value: string | undefined;
    onChange: (name: string) => void;
}

export const NameField = ({ value, onChange }: NameFieldProps) => {
    return <TextField isRequired label='Name' width='100%' value={value} onChange={onChange} />;
};
