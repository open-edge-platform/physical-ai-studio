import { Flex, Heading, Text } from '@geti-ui/ui';

import classes from './step-header.module.css';

interface StepHeaderProps {
    title: string;
    description: string;
}

export const StepHeader = ({ title, description }: StepHeaderProps) => (
    <Flex direction='column' gap='size-50'>
        <Heading level={2} margin={0}>
            {title}
        </Heading>
        <Text UNSAFE_className={classes.description}>{description}</Text>
    </Flex>
);
