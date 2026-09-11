import { useMemo } from 'react';

import { Item, Picker } from '@geti-ui/ui';

import { $api } from '../../../../api/client';

import classes from './ssh-target-fields.module.css';

export const SSH_DEVICE_TYPES = ['cuda', 'xpu'] as const;

export type SshDeviceType = (typeof SSH_DEVICE_TYPES)[number];

export const isSshDeviceType = (value: string): value is SshDeviceType =>
    (SSH_DEVICE_TYPES as readonly string[]).includes(value);

type SshTargetFieldsProps = {
    sshHostAlias: string | undefined;
    onSshHostAliasChange: (alias: string | undefined) => void;
    deviceType: SshDeviceType | undefined;
    onDeviceTypeChange: (deviceType: SshDeviceType | undefined) => void;
    deviceTypeDescription?: string;
};

/**
 * SSH host alias + device type picker pair shared by the create ("New
 * training target") and edit ("Edit SSH server") forms — same alias lookup,
 * resolved-host preview, and device type options either way.
 */
export const SshTargetFields = ({
    sshHostAlias,
    onSshHostAliasChange,
    deviceType,
    onDeviceTypeChange,
    deviceTypeDescription = 'Determines which trainer image is provisioned.',
}: SshTargetFieldsProps) => {
    const { data: aliasOptions = [] } = $api.useQuery('get', '/api/remote-servers/aliases');
    const resolvedAlias = useMemo(
        () => aliasOptions.find((option) => option.alias === sshHostAlias),
        [aliasOptions, sshHostAlias]
    );

    return (
        <>
            <Picker
                label='SSH host'
                description='Pick a Host entry from your ~/.ssh/config.'
                selectedKey={sshHostAlias ?? null}
                onSelectionChange={(key) => onSshHostAliasChange(key ? String(key) : undefined)}
                width='100%'
                items={aliasOptions}
                isRequired
            >
                {(option) => <Item key={option.alias}>{option.alias}</Item>}
            </Picker>
            {resolvedAlias !== undefined && (
                <dl className={classes.resolvedHost}>
                    <dt>Resolves to</dt>
                    <dd>
                        {[resolvedAlias.user, resolvedAlias.hostname].filter(Boolean).join('@') || '—'}
                        {resolvedAlias.port ? `:${resolvedAlias.port}` : ''}
                    </dd>
                </dl>
            )}
            <Picker
                label='Device type'
                description={deviceTypeDescription}
                selectedKey={deviceType ?? null}
                onSelectionChange={(key) => onDeviceTypeChange(key ? (String(key) as SshDeviceType) : undefined)}
                width='100%'
                isRequired
            >
                {SSH_DEVICE_TYPES.map((type) => (
                    <Item key={type}>{type.toUpperCase()}</Item>
                ))}
            </Picker>
        </>
    );
};
