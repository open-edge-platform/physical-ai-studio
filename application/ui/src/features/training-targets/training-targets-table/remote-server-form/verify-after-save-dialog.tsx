import { Button, ButtonGroup, Content, Dialog, Divider, Heading, Text, ToastQueue } from '@geti-ui/ui';

import { getApiErrorMessage } from '../../../../api/errors';
import { SchemaRemoteServer } from '../../../../api/openapi-spec';
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

    // Fire-and-forget: Tier 2 (SSH connect, registry round trips, a one-shot
    // container launch to probe the device) can take tens of seconds, and
    // this dialog has no way to show incremental progress. Closing
    // immediately instead of awaiting `onSuccess` keeps that latency from
    // blocking the user behind a spinner. The mutation keeps running after
    // this component unmounts - React Query doesn't abort in-flight
    // mutations on unmount - and the global `MutationCache` in
    // `query-client.ts` invalidates the server list once it resolves
    // regardless of which component fired it, so the training-targets table
    // (and its own "Pull & verify image" row action) picks up the real
    // `last_check_status` on its own. A failure is surfaced as a toast since
    // there's no longer a dialog around to show it inline.
    const startVerification = () => {
        checkMutation.mutate(
            { params: { path: { remote_server_id: savedServer.id } } },
            {
                onError: (error) => {
                    const fallback = `'${savedServer.name}': image pull/verification failed.`;
                    const hint = 'Try again from the training-targets table.';
                    ToastQueue.negative(getApiErrorMessage(error) ?? `${fallback} ${hint}`);
                },
            }
        );
        close();
    };

    return (
        <Dialog>
            <Heading>Pull &amp; verify trainer image?</Heading>
            <Divider />
            <Content>
                <Text>
                    {`'${savedServer.name}' is saved. Pull and verify the trainer image now to start the download `}
                    {'of the trainer image. '}
                    Skipping it means you&apos;ll need to verify the server, from the training-targets table or when
                    training, before you can submit a job to it.
                </Text>
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={close}>
                    Skip for now
                </Button>
                <Button variant='accent' onPress={startVerification}>
                    Pull &amp; verify image
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};
