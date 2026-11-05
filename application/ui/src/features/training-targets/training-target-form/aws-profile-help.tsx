import { ActionButton, Content, Dialog, DialogTrigger, Heading, Link } from '@geti-ui/ui';
import { InfoOutline } from '@geti-ui/ui/icons';

export const AwsProfileHelp = () => (
    <DialogTrigger type='popover'>
        <ActionButton type='button' isQuiet aria-label='Named profile help'>
            <InfoOutline />
        </ActionButton>
        <Dialog width='size-6000'>
            <Heading>Named profile</Heading>
            <Content>
                <p>Configure an AWS named profile on the machine running the Studio backend.</p>
                <p>
                    For IAM Identity Center (SSO), run <code>aws configure sso --profile studio</code>, then{' '}
                    <code>aws sso login --profile studio</code>. For access keys, run{' '}
                    <code>aws configure --profile studio</code>.
                </p>
                <p>
                    With Docker, make your <code>~/.aws</code> configuration, credentials, and session cache accessible
                    to the backend container user.
                </p>
                <p>
                    Select the profile in this dropdown, then connect to the stack. Reopen the dropdown to refresh the
                    list after setup. When credentials expire, renew the profile on the backend host and retry.
                </p>
                <p>
                    The profile needs permission to read the S3 configuration file and assume the stack’s Studio role.
                    The role must trust this identity.
                </p>
                <Link
                    href={
                        'https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html' +
                        '#cli-configure-files-using-profiles'
                    }
                    target='_blank'
                    rel='noopener noreferrer'
                >
                    Configure a named profile
                </Link>
            </Content>
        </Dialog>
    </DialogTrigger>
);
