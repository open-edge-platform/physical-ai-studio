import { Text } from '@geti/ui';
import { AddCircle } from '@geti/ui/icons';
import { Link } from 'react-router-dom';

import { paths } from '../../../router';

import classes from './project-list.module.scss';

export const NewProjectLink = () => {
    return (
        <Link to={paths.projects.new.pattern} className={classes.link}>
            <AddCircle />
            <Text>Add another project</Text>
        </Link>
    );
};
