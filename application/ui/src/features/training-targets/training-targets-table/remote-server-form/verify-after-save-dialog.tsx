import { Button, ButtonGroup, Content, Dialog, Divider, Flex, Heading, Text } from '@geti-ui/ui';

import { getApiErrorMessage } from '../../../../api/errors';
import { SchemaRemoteServer } from '../../../../api/openapi-spec';
import { InlineAlert } from '../../../robots/setup-wizard/shared/inline-alert';
import { useRemoteServerCheckMutation } from '../use-remote-server-check-mutation';

type VerifyAfterSaveDialogProps = {
    savedServer: SchemaRemoteServer;
    close: () => void;
};

/**
 * Prompts to pull & verify the trainer image right after a *new* SSH server is saved.
 *
 * A freshly saved server's `last_check_status` is always `"unknown"` - nobody
 * has run the (multi-gigabyte, one-shot-container) Tier 2 check yet. Offering
 * it here, once, means the user doesn't have to separately discover the
 * "Pull & verify image" button on the training-targets table to get the
 * server ready for a job.
 *
 * Skipping is allowed, but it is *not* a no-op deferral: the train-model
 * dialog treats any `last_check_status !== "healthy"` (including
 * `"unknown"`) as not ready and disables job submission until the server is
 * verified, either from this dialog, the "Pull & verify image" button on the
 * training-targets table, or the train-model dialog itself. The backend's
 * `services.training_targets.ssh.SshTrainingTargetHandler.prepare` does
 * auto-verify a never-checked server, but the UI never reaches that code
 * path since it blocks submission first - so don't describe skipping as
 * something job submission "handles for you".
 */
export const VerifyAfterSaveDialog = ({ savedServer, close }: VerifyAfterSaveDialogProps) => {
    const checkMutation = useRemoteServerCheckMutation();
    const errorMessage = checkMutation.isError
        ? (getApiErrorMessage(checkMutation.error) ?? 'The image could not be pulled or verified. Try again.')
        : undefined;

    return (
        <Dialog>
            <Heading>Pull &amp; verify trainer image?</Heading>
            <Divider />
            <Content>
                <Flex direction='column' gap='size-200'>
                    <Text>
                        {`'${savedServer.name}' is saved. Pull and verify the trainer image now to start the download `}
                        {'of the trainer image. '}
                        Skipping it means you&apos;ll need to verify the server, from the training-targets table or when
                        training, before you can submit a job to it.
                    </Text>
                    {errorMessage !== undefined && <InlineAlert variant='error'>{errorMessage}</InlineAlert>}
                </Flex>
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={close} isDisabled={checkMutation.isPending}>
                    Skip for now
                </Button>
                <Button
                    variant='accent'
                    isPending={checkMutation.isPending}
                    onPress={() =>
                        checkMutation.mutate(
                            { params: { path: { remote_server_id: savedServer.id } } },
                            { onSuccess: close }
                        )
                    }
                >
                    Pull &amp; verify image
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};
