import { useState } from 'react';

import { AlertDialog, DialogContainer, Flex, Heading, Key, Text, View } from '@geti-ui/ui';
import { clsx } from 'clsx';
import { NavLink } from 'react-router';

import { $api } from '../../../api/client';
import { SchemaProjectInput } from '../../../api/openapi-spec';
import { paths } from '../../../router';
import { ProjectThumbnail } from '../project-thumbnail/project-thumbnail';
import { MenuActions } from './menu-actions.component';

import classes from './project-list.module.css';

const IMAGE_SIZE = 132;

type ProjectCardProps = {
    item: SchemaProjectInput;
    isActive: boolean;
};

export const ProjectCard = ({ item, isActive }: ProjectCardProps) => {
    const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);
    const deleteMutation = $api.useMutation('delete', '/api/projects/{project_id}', {
        meta: {
            invalidates: [['get', '/api/projects']],
        },
    });

    const deleteProject = () => {
        deleteMutation.mutate({
            params: { path: { project_id: item.id } },
        });
    };

    const closeDeleteDialog = () => {
        setIsDeleteDialogOpen(false);
    };

    const handleActions = (key: Key) => {
        switch (key.toString()) {
            case 'delete':
                setIsDeleteDialogOpen(true);
                break;
            default:
                break;
        }
    };

    return (
        <>
            <NavLink to={paths.project.robots.index({ project_id: item.id! })}>
                <Flex UNSAFE_className={clsx(classes.card, { [classes.activeCard]: isActive })}>
                    <View aria-label={'project thumbnail'} UNSAFE_className={classes.imgWrapper}>
                        <ProjectThumbnail projectId={item.id!} name={item.name} size={IMAGE_SIZE} />
                    </View>

                    <View width={'100%'} padding={'size-200'}>
                        <Flex alignItems={'center'} justifyContent={'space-between'}>
                            <Heading level={3}>{item.name}</Heading>
                            <MenuActions onAction={handleActions} />
                        </Flex>

                        <Flex
                            alignItems={'start'}
                            gap={'size-100'}
                            direction={'column'}
                            wrap='wrap'
                            marginTop='size-100'
                        >
                            {item.updated_at !== undefined && (
                                <Text>• Edited: {new Date(item.updated_at!).toLocaleString()}</Text>
                            )}
                            <Flex alignItems={'center'} gap={'size-100'} direction={'row'} wrap='wrap'>
                                {item.datasets.length > 0 && (
                                    <Text>• Datasets: {item.datasets.map((d) => d.name).join(', ')}</Text>
                                )}
                            </Flex>
                        </Flex>
                    </View>
                </Flex>
            </NavLink>
            <DialogContainer onDismiss={closeDeleteDialog}>
                {isDeleteDialogOpen && (
                    <AlertDialog
                        title={'Delete project'}
                        primaryActionLabel={'Delete'}
                        variant={'destructive'}
                        onPrimaryAction={deleteProject}
                        isPrimaryActionDisabled={deleteMutation.isPending}
                        secondaryActionLabel={'Cancel'}
                        onSecondaryAction={closeDeleteDialog}
                    >
                        <Text>{`Are you sure you want to delete the project "${item.name}"?`}</Text>
                    </AlertDialog>
                )}
            </DialogContainer>
        </>
    );
};
