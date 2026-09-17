import { Button, ButtonGroup, Content, Dialog, Divider, Flex, Heading, Icon, Text } from '@geti-ui/ui';
import { AlertCircle } from '@geti-ui/ui/icons';

import classes from './ssh-host-key-confirmation-dialog.module.css';

export type SshHostKeyConfirmation = {
    host: string;
    fingerprint: string;
    onConfirm: () => void;
    onCancel: () => void;
};

type SshHostKeyConfirmationDialogProps = Pick<SshHostKeyConfirmation, 'host' | 'fingerprint'> & {
    onConfirm: () => void;
    onCancel: () => void;
};

export const SshHostKeyConfirmationDialog = ({
    host,
    fingerprint,
    onConfirm,
    onCancel,
}: SshHostKeyConfirmationDialogProps) => (
    <Dialog width='size-5000' UNSAFE_className={classes.dialog}>
        <Heading UNSAFE_className={classes.heading}>
            <Icon UNSAFE_className={classes.warningIcon}>
                <AlertCircle />
            </Icon>
            Verify SSH server
        </Heading>
        <Divider />
        <Content>
            <Flex direction='column' gap='size-200'>
                <Text>
                    Studio has not connected to this server before. Confirm that this fingerprint matches the server
                    before trusting it.
                </Text>
                <div className={classes.identity}>
                    <Text UNSAFE_className={classes.label}>SSH server</Text>
                    <Text UNSAFE_className={classes.host}>{host}</Text>
                    <Text UNSAFE_className={classes.label}>Fingerprint</Text>
                    <code className={classes.fingerprint}>{fingerprint}</code>
                </div>
            </Flex>
        </Content>
        <ButtonGroup UNSAFE_className={classes.actions}>
            <Button variant='secondary' onPress={onCancel}>
                Cancel
            </Button>
            <Button variant='negative' onPress={onConfirm}>
                Trust host
            </Button>
        </ButtonGroup>
    </Dialog>
);
