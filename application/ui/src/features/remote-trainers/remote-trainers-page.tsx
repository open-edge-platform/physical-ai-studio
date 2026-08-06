import { useState } from 'react';

import { Button, Flex, Heading, Text, View } from '@geti-ui/ui';
import { Add } from '@geti-ui/ui/icons';

import { $api } from '../../api/client';
import { RemoteTrainerAction, RemoteTrainersTable } from './remote-trainers-table/remote-trainers-table';

import classes from './remote-trainers-page.module.css';

export const RemoteTrainersPage = () => {
    const { data: remoteTrainers } = $api.useSuspenseQuery('get', '/api/remote-trainers');
    const [action, setAction] = useState<RemoteTrainerAction>();

    return (
        <View padding='size-400' height='100%' maxWidth='240ch' marginX='auto'>
            <Flex marginBottom={'size-250'} justifyContent={'space-between'} alignItems={'center'}>
                <View>
                    <Heading level={1}>Remote Trainers</Heading>
                    <Text>Configure and monitor trainer endpoints for remote training jobs.</Text>
                </View>

                <Button
                    variant='accent'
                    UNSAFE_className={classes.addButton}
                    onPress={() => setAction({ type: 'create' })}
                >
                    <Add />
                    <Text>New remote trainer</Text>
                </Button>
            </Flex>

            <View UNSAFE_className={classes.table}>
                {remoteTrainers.length === 0 ? (
                    <Text UNSAFE_className={classes.emptyList}>No remote trainers are configured.</Text>
                ) : (
                    <RemoteTrainersTable action={action} setAction={setAction} remoteTrainers={remoteTrainers} />
                )}
            </View>
        </View>
    );
};
