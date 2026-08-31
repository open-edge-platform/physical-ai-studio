import { ReactNode, useState } from 'react';

import { AlertDialog, Button, DialogContainer, Flex, Heading, Text, View } from '@geti-ui/ui';

import { $api } from '../../../api/client';
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
    /**
     * When set, saving no longer happens directly: the user is shown a confirmation dialog with this
     * message first, and `onSave` only runs if they confirm. Use this for settings whose side effects
     * (e.g. a backend restart) aren't obvious from the form itself. The dialog also warns separately if
     * any jobs are currently running, since a restart will interrupt them.
     */
    restartWarning?: string;
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
    restartWarning,
}: SettingsSectionProps) => {
    const [isConfirmOpen, setConfirmOpen] = useState(false);

    // Only fetched while the confirmation dialog can be shown, to warn about interrupting active jobs.
    const { data: jobs } = $api.useQuery('get', '/api/jobs', undefined, { enabled: restartWarning !== undefined });
    const runningJobCount = jobs?.filter((job) => job.status === 'running').length ?? 0;

    const errorMessage = error
        ? (getApiErrorMessage(error) ?? 'The settings could not be saved. Try again.')
        : undefined;

    const handleSavePress = () => {
        if (restartWarning !== undefined) {
            setConfirmOpen(true);
            return;
        }
        onSave();
    };

    const confirmSave = () => {
        setConfirmOpen(false);
        onSave();
    };

    return (
        <View UNSAFE_className={classes.section} padding='size-300'>
            <Heading level={3}>{title}</Heading>
            <Text UNSAFE_className={classes.description}>{description}</Text>
            <Flex direction='column' gap='size-200' UNSAFE_className={classes.fields}>
                {children}
            </Flex>
            {errorMessage !== undefined && <Text UNSAFE_className={classes.error}>{errorMessage}</Text>}
            <Flex alignItems='center' gap='size-200' marginTop='size-200'>
                <Button
                    variant='accent'
                    onPress={handleSavePress}
                    isDisabled={!isDirty || isPending}
                    isPending={isPending}
                >
                    Save
                </Button>
                {saved && !isDirty && <Text UNSAFE_className={classes.saved}>Saved</Text>}
            </Flex>
            <DialogContainer onDismiss={() => setConfirmOpen(false)}>
                {isConfirmOpen && (
                    <AlertDialog
                        title='Restart required'
                        variant={runningJobCount > 0 ? 'warning' : 'information'}
                        primaryActionLabel='Save and restart'
                        cancelLabel='Cancel'
                        onPrimaryAction={confirmSave}
                        onCancel={() => setConfirmOpen(false)}
                    >
                        <Flex direction='column' gap='size-150'>
                            <Text>{restartWarning}</Text>
                            {runningJobCount > 0 && (
                                <Text>
                                    <strong>
                                        {runningJobCount} job{runningJobCount > 1 ? 's are' : ' is'} currently running.
                                    </strong>{' '}
                                    Restarting the backend will interrupt {runningJobCount > 1 ? 'them' : 'it'}. Make
                                    sure that&apos;s okay before continuing.
                                </Text>
                            )}
                        </Flex>
                    </AlertDialog>
                )}
            </DialogContainer>
        </View>
    );
};
