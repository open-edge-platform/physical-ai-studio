import { ReactNode } from 'react';

import { Flex, Text } from '@geti-ui/ui';
import { Add } from '@geti-ui/ui/icons';
import { Link } from 'react-router';

import classes from './add-resource-button.module.css';

interface AddResourceButtonProps {
    to: string;
    children: ReactNode;
}

export const AddResourceButton = ({ to, children }: AddResourceButtonProps) => {
    return (
        <Link to={to} className={classes.addResourceButton}>
            <Flex alignItems='center' justifyContent='center' gap='size-75' width={'100%'}>
                <Add />
                <Text UNSAFE_style={{ lineHeight: '1.2' }}>{children}</Text>
            </Flex>
        </Link>
    );
};
