import { Content, ContextualHelp, Footer, Heading, Link, Text } from '@geti-ui/ui';

import { ContextualInfo } from '../types';

type FieldContextualHelpProps = {
    info?: ContextualInfo;
};

export const FieldContextualHelp = ({ info }: FieldContextualHelpProps) => {
    if (info === undefined) {
        return null;
    }

    return (
        <ContextualHelp variant={info.variant ?? 'info'}>
            {info.title !== undefined && info.title !== '' && <Heading>{info.title}</Heading>}
            <Content>
                <Text>{info.description}</Text>
            </Content>
            {info.link_url !== undefined && info.link_url !== '' && (
                <Footer>
                    <Link href={info.link_url} target='_blank' rel='noopener noreferrer'>
                        Learn more
                    </Link>
                </Footer>
            )}
        </ContextualHelp>
    );
};
