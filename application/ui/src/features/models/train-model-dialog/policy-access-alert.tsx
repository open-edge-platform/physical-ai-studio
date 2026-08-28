import { Flex, Link as SpectrumLink } from '@geti-ui/ui';
import { Link } from 'react-router';

import { $api } from '../../../api/client';
import { paths } from '../../../router';
import { InlineAlert } from '../../robots/setup-wizard/shared/inline-alert';

type PolicyAccessAlertProps = { policy: string };

export const PolicyAccessAlert = ({ policy }: PolicyAccessAlertProps) => {
    const { data: policyAccess } = $api.useQuery('get', '/api/policies/{policy}/huggingface-access', {
        params: { path: { policy } },
    });

    const failedRequirements =
        policyAccess?.requirements.filter(({ status }) => status === 'missing_token' || status === 'denied') ?? [];
    const blocksTraining = failedRequirements.some(({ required }) => required);

    if (failedRequirements.length > 0) {
        return (
            <InlineAlert variant={blocksTraining ? 'error' : 'warning'}>
                <Flex direction='column' gap='size-100'>
                    <span>
                        {failedRequirements.some((requirement) => requirement.status === 'missing_token')
                            ? 'This policy downloads pretrained assets from Hugging Face, but no token is configured.'
                            : 'Your Hugging Face token does not have access to this policy.'}
                    </span>
                    {failedRequirements.map((requirement) => (
                        <SpectrumLink
                            key={requirement.repository}
                            href={requirement.access_url}
                            target='_blank'
                            rel='noopener noreferrer'
                        >
                            Request access to {requirement.repository}
                        </SpectrumLink>
                    ))}
                    <Link to={paths.settings.index.pattern}>Open Settings</Link>
                </Flex>
            </InlineAlert>
        );
    }

    const unavailable = policyAccess?.requirements.some((requirement) => requirement.status === 'unavailable');
    if (unavailable) {
        return (
            <InlineAlert variant='warning'>
                We couldn&apos;t verify Hugging Face access. Check your network connection and try again.
            </InlineAlert>
        );
    }

    return null;
};
