import { useId } from 'react';

import { Flex, Item, Picker, Text } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { IssueErrorAlert } from '../../../components/notification/issue-toasts';
import { AwsProfileHelp } from './aws-profile-help';

import classes from '../training-targets-table/remote-trainer-form/remote-trainer-form.module.css';

interface AwsCredentialsProps {
    profileName: string;
    onProfileChange: (value: string) => void;
    isDisabled: boolean;
}

export const AwsCredentials = ({ profileName, onProfileChange, isDisabled }: AwsCredentialsProps) => {
    const descriptionId = useId();
    const profiles = $api.useQuery('get', '/api/remote-trainers/providers/aws/auth');
    const error = profiles.error;
    const profileOptions = [...new Set([...(profiles.data?.profiles ?? []), ...(profileName ? [profileName] : [])])];

    return (
        <Flex direction='column' gap='size-100'>
            <Flex gap='size-100' alignItems='end'>
                <Picker
                    label='AWS profile'
                    aria-describedby={descriptionId}
                    isRequired
                    selectedKey={profileName || null}
                    onSelectionChange={(key) => {
                        if (key !== null) {
                            onProfileChange(String(key));
                        }
                    }}
                    onOpenChange={(open) => {
                        if (open) void profiles.refetch();
                    }}
                    isDisabled={isDisabled || profiles.isPending}
                    flex
                >
                    {profileOptions.map((name) => (
                        <Item key={name}>{name}</Item>
                    ))}
                </Picker>
                <AwsProfileHelp />
            </Flex>
            <Text id={descriptionId} UNSAFE_className={classes.helperText}>
                Configure and authenticate this profile on the Studio backend host before selecting it. Studio uses its
                credentials to access your training stack.
            </Text>
            {profiles.data?.profiles?.length === 0 && !error && (
                <div role='status'>No AWS profiles found. Open the information hint for setup instructions.</div>
            )}
            {error && <IssueErrorAlert error={error} />}
        </Flex>
    );
};
