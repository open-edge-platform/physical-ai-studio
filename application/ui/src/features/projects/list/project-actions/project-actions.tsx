import { Button, Flex, Text, TextField } from '@geti-ui/ui';
import { Add } from '@geti-ui/ui/icons';

import { pluralize } from '../../../../utils';
import { CreateProject } from '../no-projects/create-project';

import classes from './project-actions.module.css';

type ProjectActionsProps = {
    totalProjectsCount: number;
    projectsCount: number;
    onSearch: (query: string) => void;
    searchQuery: string;
};

export const ProjectActions = ({ totalProjectsCount, projectsCount, searchQuery, onSearch }: ProjectActionsProps) => {
    const hasFilters = searchQuery.length > 0;
    const projectsText = pluralize(totalProjectsCount, 'project', 'projects');

    return (
        <Flex justifyContent={'end'} gap={'size-200'} alignItems={'center'}>
            <Text UNSAFE_className={classes.countMessage}>
                {hasFilters && totalProjectsCount !== projectsCount
                    ? `${projectsCount} of ${totalProjectsCount} ${projectsText}`
                    : `${totalProjectsCount} ${projectsText}`}
            </Text>
            <TextField
                flex={1}
                value={searchQuery}
                onChange={onSearch}
                placeholder={'Search...'}
                aria-label={'Search projects'}
            />
            <CreateProject
                trigger={
                    <Button variant={'primary'}>
                        <Add style={{ fill: 'var(--spectrum-global-color-gray-900)' }} />
                        <Text marginStart={'size-50'}>Create new project</Text>
                    </Button>
                }
            />
        </Flex>
    );
};
