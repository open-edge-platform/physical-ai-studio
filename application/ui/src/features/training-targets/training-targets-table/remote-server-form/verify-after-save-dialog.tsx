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
 * server ready ahead of time.
 *
 * Skipping is allowed and *is* a real no-op deferral: the train-model dialog
 * lets a job submit against an `"unknown"` server (see
 * `TrainModelDialog`'s `sshUnverified`), and the backend's
 * `services.training_targets.ssh.SshTrainingTargetHandler.prepare` runs the
 * same Tier 2 verification automatically the first time a job actually needs
 * it. This dialog only exists to let the user front-load that latency
 * (pulling a multi-gigabyte image) instead of hitting it in the middle of
 * submitting a job; the server works either way, once verified one way or
 * the other.
 */
export const VerifyAfterSaveDialog = ({ savedServer, close }: VerifyAfterSaveDialogProps) => {
    const checkMutation = useRemoteServerCheckMutation();

    // Fire-and-forget: Tier 2 (SSH connect, registry round trips, a one-shot
    // container launch to probe the device) can take tens of seconds, and
    // this dialog has no way to show incremental progress. Closing
    // immediately keeps that latency from blocking the user behind a
    // spinner. The mutation keeps running after this component unmounts -
    // React Query doesn't abort in-flight mutations on unmount - and the
    // global `MutationCache` in `query-client.ts` invalidates the server list
    // once it resolves regardless of which component fired it, so the
    // training-targets table (and its own "Pull & verify image" row action)
    // picks up the real `last_check_status` on its own. A failure is
    // surfaced as a toast since there's no longer a dialog around to show it
    // inline - handled off `mutateAsync`'s own promise rather than the
    // `mutate(...)`-time `onError` option, since that option is delivered by
    // the mutation *observer* and is silently dropped once this component (and
    // its observer) unmounts, which `close()` does immediately below.
    const startVerification = () => {
        checkMutation.mutateAsync({ params: { path: { remote_server_id: savedServer.id } } }).catch((error) => {
            const fallback = `'${savedServer.name}': image pull/verification failed.`;
            const hint = 'Try again from the training-targets table.';
            ToastQueue.negative(getApiErrorMessage(error) ?? `${fallback} ${hint}`);
        });
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
