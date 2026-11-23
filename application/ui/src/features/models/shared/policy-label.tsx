import { Flex, Text } from '@geti-ui/ui';

import { SchemaModel } from '../../../api/openapi-spec';

export const PolicyLabel = ({ policy, reference }: { policy: string; reference?: SchemaModel['policy_reference'] }) => (
    <Flex direction='column' gap='size-50'>
        <Text>{policy.toUpperCase()}</Text>
        {reference?.status === 'unavailable' && <span title={reference.reason}>Policy unavailable</span>}
        {reference?.status === 'unknown' && <span title={reference.reason}>Availability unknown</span>}
    </Flex>
);
