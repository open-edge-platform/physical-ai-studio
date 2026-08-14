import { useState } from 'react';

import { Flex, Heading, Key, Text, View } from '@geti-ui/ui';
import { clsx } from 'clsx';
import { NavLink } from 'react-router';

import { $api, fetchClient } from '../../../api/client';
import { SchemaProjectInput } from '../../../api/openapi-spec';
import { paths } from '../../../router';
import { ReactComponent as PhysicalAIStudioLogo } from './../../../assets/icons/physicalai-studio-logo.svg';
import { MenuActions } from './menu-actions.component';

import classes from './project-list.module.css';

const IMAGE_SIZE = 132;

type ProjectCardProps = {
    item: SchemaProjectInput;
    isActive: boolean;
};

export const ProjectCard = ({ item, isActive }: ProjectCardProps) => {
    const [hasThumbnailError, setHasThumbnailError] = useState(false);
    const deleteMutation = $api.useMutation('delete', '/api/projects/{project_id}', {
        meta: {
            invalidates: [['get', '/api/projects']],
        },
    });

    const projectThumbnailUrl = fetchClient.PATH('/api/projects/{project_id}/thumbnail', {
        params: { path: { project_id: item.id! }, query: { width: IMAGE_SIZE, height: IMAGE_SIZE } },
    });

    const onAction = (key: Key) => {
        switch (key.toString()) {
            case 'delete':
                if (item.id !== undefined) {
                    deleteMutation.mutate({
                        params: { path: { project_id: item.id } },
                    });
                }
                return;
        }
    };

    return (
        <NavLink to={paths.project.robots.index({ project_id: item.id! })}>
            <Flex UNSAFE_className={clsx({ [classes.card]: true, [classes.activeCard]: isActive })}>
                <View aria-label={'project thumbnail'} UNSAFE_className={classes.imgWrapper}>
                    {hasThumbnailError ? (
                        <View width={IMAGE_SIZE} height={IMAGE_SIZE} backgroundColor={'gray-100'}>
                            <Flex justifyContent={'center'} alignItems={'center'} width={'100%'} height={'100%'}>
                                <PhysicalAIStudioLogo width={80} height={80} style={{ filter: 'grayscale(100)' }} />
                            </Flex>
                        </View>
                    ) : (
                        <img
                            style={{ width: IMAGE_SIZE, height: IMAGE_SIZE }}
                            src={projectThumbnailUrl}
                            alt={item.name}
                            onError={() => setHasThumbnailError(true)}
                        />
                    )}
                </View>

                <View width={'100%'} padding={'size-200'}>
                    <Flex alignItems={'center'} justifyContent={'space-between'}>
                        <Heading level={3}>{item.name}</Heading>
                        <MenuActions onAction={onAction} />
                    </Flex>

                    <Flex alignItems={'start'} gap={'size-100'} direction={'column'} wrap='wrap' marginTop='size-100'>
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
    );
};
