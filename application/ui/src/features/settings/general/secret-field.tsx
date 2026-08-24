import { useState } from 'react';

import { Flex, Switch, TextField } from '@geti-ui/ui';

export type SecretChange = { draft: string; remove: boolean };

type SecretFieldProps = {
    label: string;
    isSet: boolean;
    onChange: (change: SecretChange) => void;
};

export const SecretField = ({ label, isSet, onChange }: SecretFieldProps) => {
    const [draft, setDraft] = useState('');
    const [remove, setRemove] = useState(false);

    const update = (nextDraft: string, nextRemove: boolean) => {
        setDraft(nextDraft);
        setRemove(nextRemove);
        onChange({ draft: nextDraft, remove: nextRemove });
    };

    return (
        <Flex direction='column' gap='size-100'>
            <TextField
                type='password'
                label={label}
                value={draft}
                onChange={(value) => update(value, remove)}
                placeholder={isSet ? 'Leave empty to keep the configured value' : undefined}
                width='100%'
            />
            {isSet && (
                <Switch isSelected={remove} onChange={(selected) => update(draft, selected)}>
                    Remove the configured value
                </Switch>
            )}
        </Flex>
    );
};
