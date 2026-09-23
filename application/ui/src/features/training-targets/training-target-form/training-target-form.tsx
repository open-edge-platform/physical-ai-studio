import { RemoteTrainerForm } from '../training-targets-table/remote-trainer-form/remote-trainer-form';
import { SshHostKeyConfirmation } from '../training-targets-table/ssh-host-key-confirmation-dialog';

type TrainingTargetFormProps = {
    close: () => void;
    requestHostKeyConfirmation: (confirmation: SshHostKeyConfirmation) => void;
};

/** New targets are remote trainers; SSH is configured through its SSH tunnel tab. */
export const TrainingTargetForm = ({ close, requestHostKeyConfirmation }: TrainingTargetFormProps) => (
    <RemoteTrainerForm close={close} requestHostKeyConfirmation={requestHostKeyConfirmation} />
);
