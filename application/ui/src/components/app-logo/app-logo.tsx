import { Flex, Link } from '@geti-ui/ui';

import { ReactComponent as PhysicalAIStudioLogo } from '../../assets/icons/physicalai-studio-logo.svg';
import { paths } from '../../router';

export const AppLogo = () => {
    return (
        <Link href={paths.projects.index({})} isQuiet variant='overBackground' marginEnd='size-200'>
            <Flex gap='size-200' alignItems={'center'}>
                <PhysicalAIStudioLogo />
                <span style={{ whiteSpace: 'nowrap', fontWeight: 'bold', textDecoration: 'none' }}>
                    Physical AI Studio
                </span>
            </Flex>
        </Link>
    );
};
