import { Heading, Text } from '@geti-ui/ui';

import classes from './projects-heading.module.css';

export const ProjectsHeading = () => {
    return (
        <Heading UNSAFE_className={classes.heading}>
            Bring Robots to Life
            <br />
            <Text UNSAFE_className={classes.notBold}> with </Text>
            <Text UNSAFE_className={classes.emphasizedHeading}>Physical AI Studio</Text>
        </Heading>
    );
};
