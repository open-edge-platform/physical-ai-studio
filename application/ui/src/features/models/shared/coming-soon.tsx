import { Heading, IllustratedMessage } from '@geti-ui/ui';

import { ReactComponent as EmptyIllustration } from './../../../assets/illustration.svg';

export const ComingSoon = () => {
    return (
        <IllustratedMessage marginY='size-400'>
            <EmptyIllustration height='250px' />
            <Heading>Coming soon</Heading>
        </IllustratedMessage>
    );
};
