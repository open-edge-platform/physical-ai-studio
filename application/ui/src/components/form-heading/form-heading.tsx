import { ReactNode } from 'react';

import { Flex, Heading } from '@geti-ui/ui';
import { ChevronLeft } from '@geti-ui/ui/icons';
import { Link } from 'react-router';

import classes from './form-heading.module.css';

interface FormHeadingProps {
    heading: ReactNode;
    backTo: string;
    backLabel: string;
}

export const FormHeading = ({ heading, backTo, backLabel }: FormHeadingProps) => {
    return (
        <Flex alignItems='center' gap='size-200'>
            <Link className={classes.link} aria-label={backLabel} to={backTo}>
                <ChevronLeft color='white' fill='white' />
            </Link>
            <Heading>{heading}</Heading>
        </Flex>
    );
};
