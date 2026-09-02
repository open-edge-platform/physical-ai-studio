import { ReactNode } from 'react';

import { Button, Flex, Heading, Text, View } from '@geti-ui/ui';

import { getApiErrorMessage } from '../../../api/errors';

import classes from './general-settings.module.css';

type SettingsSectionProps = {
    title: string;
    description: string;
    isDirty: boolean;
    isPending: boolean;
    saved: boolean;
    error?: unknown;
    onSave: () => void;
    children: ReactNode;
};

export const SettingsSection = ({
    title,
    description,
    isDirty,
    isPending,
    saved,
    error,
    onSave,
    children,
}: SettingsSectionProps) => {
    const errorMessage = error
        ? (getApiErrorMessage(error) ?? 'The settings could not be saved. Try again.')
        : undefined;

    return (
        <View UNSAFE_className={classes.section} padding='size-300'>
            <Heading level={3}>{title}</Heading>
            <Text UNSAFE_className={classes.description}>{description}</Text>
            <Flex direction='column' gap='size-200' UNSAFE_className={classes.fields}>
                {children}
            </Flex>
            {errorMessage !== undefined && <Text UNSAFE_className={classes.error}>{errorMessage}</Text>}
            <Flex alignItems='center' gap='size-200' marginTop='size-200'>
                <Button variant='accent' onPress={onSave} isDisabled={!isDirty || isPending} isPending={isPending}>
                    Save
                </Button>
                {saved && !isDirty && <Text UNSAFE_className={classes.saved}>Saved</Text>}
            </Flex>
        </View>
    );
};
