import { Flex, Heading, Tag, Text, View } from '@geti/ui';
import { clsx } from 'clsx';
import { NavLink } from 'react-router-dom';

import { SchemaProject } from '../../../api/openapi-spec';
import thumbnailUrl from '../../../assets/mocked-project-thumbnail.png';
import { paths } from '../../../router';
import { MenuActions } from './menu-actions.component';

import classes from './project-list.module.scss';

type ProjectCardProps = {
    item: SchemaProject;
    isActive: boolean;
};

export const ProjectCard = ({ item, isActive }: ProjectCardProps) => {
    return (
        <NavLink to={paths.project.robotConfiguration({ project_id: item.id! })}>
            <Flex UNSAFE_className={clsx({ [classes.card]: true, [classes.activeCard]: isActive })}>
                <View aria-label={'project thumbnail'}>
                    <img src={thumbnailUrl} alt={item.name} />
                </View>

                <View width={'100%'} padding={'size-200'}>
                    <Flex alignItems={'center'} justifyContent={'space-between'}>
                        <Heading level={3}>{item.name}</Heading>
                        <MenuActions />
                    </Flex>

                    <Flex marginBottom={'size-200'} gap={'size-50'}>
                        {isActive && (
                            <Tag withDot={false} text='Active' className={clsx(classes.tag, classes.activeTag)} />
                        )}
                        <Tag withDot={false} text={item.name} className={classes.tag} />
                    </Flex>

                    <Flex alignItems={'center'} gap={'size-100'} direction={'row'} wrap='wrap'>
                        <Text>• Edited: 2025-08-07 06:05 AM</Text>
                        <Text>• Datasets: </Text>
                        <Text>• Cameras: </Text>
                        <Text>
                            • Robots: 
                        </Text>
                    </Flex>
                </View>
            </Flex>
        </NavLink>
    );
};
