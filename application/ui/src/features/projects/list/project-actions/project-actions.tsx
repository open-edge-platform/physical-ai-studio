import { Dispatch, SetStateAction } from 'react';

import { ActionButton, Button, Flex, Text, TextField, View } from '@geti-ui/ui';
import { Add, SortUp } from '@geti-ui/ui/icons';

import { pluralize } from '../../../../utils';
import { CreateProject } from '../no-projects/create-project';

import classes from './project-actions.module.css';

export type SortDirection = 'asc' | 'desc';

type ProjectActionsProps = {
    totalProjectsCount: number;
    projectsCount: number;
    onSearch: (query: string) => void;
    searchQuery: string;
    sortDirection: SortDirection;
    onSortDirectionChange: Dispatch<SetStateAction<SortDirection>>;
};

export const ProjectActions = ({
    totalProjectsCount,
    projectsCount,
    searchQuery,
    onSearch,
    sortDirection,
    onSortDirectionChange,
}: ProjectActionsProps) => {
    const hasFilters = searchQuery.length > 0;
    const projectsText = pluralize(totalProjectsCount, 'project', 'projects');

    return (
        <Flex justifyContent={'end'} gap={'size-200'} alignItems={'center'}>
            <View width={'size-1250'}>
                <ActionButton
                    isQuiet
                    UNSAFE_className={classes.sortButton}
                    onPress={() => onSortDirectionChange((prev) => (prev === 'asc' ? 'desc' : 'asc'))}
                >
                    {sortDirection === 'desc' ? 'Newest first' : 'Oldest first'}
                    <SortUp
                        style={{ fill: 'var(--spectrum-global-color-gray-900)' }}
                        className={[classes.sortIcon, sortDirection === 'desc' ? classes.sortIconDown : ''].join(' ')}
                    />
                </ActionButton>
            </View>
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
