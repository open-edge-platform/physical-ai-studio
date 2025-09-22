import {
    ActionButton,
    ActionMenu,
    ButtonGroup,
    Content,
    Dialog,
    DialogTrigger,
    Divider,
    Flex,
    Header,
    Heading,
    Icon,
    Item,
    Link,
    Menu,
    PhotoPlaceholder,
    Text,
    View,
} from '@geti/ui';
import { AddCircle, ChevronRightSmallLight } from '@geti/ui/icons';

import { $api } from '../../../api/client';
import { paths } from '../../../router';
import { useProjectId } from './../use-project';

import styles from './projects-list.module.scss';

interface Project {
    name: string;
    id: string;
}
interface ProjectListProps {
    menuWidth?: string;
    projects: Project[];
}

export const ProjectsList = ({ menuWidth = '100%', projects }: ProjectListProps) => {
    return (
        <Menu UNSAFE_className={styles.projectMenu} width={menuWidth} items={projects}>
            {(item) => (
                <Item
                    key={item.id}
                    textValue={item.name}
                    href={paths.project.robotConfiguration({ project_id: item.id })}
                >
                    <Text>
                        <Flex justifyContent='space-between' alignItems='center' marginX={'size-200'}>
                            {item.name}
                            <ActionMenu isQuiet>
                                <Item>Rename</Item>
                                <Item>Delete</Item>
                            </ActionMenu>
                        </Flex>
                    </Text>
                </Item>
            )}
        </Menu>
    );
};

interface SelectedProjectProps {
    name: string;
}

const SelectedProjectButton = ({ name }: SelectedProjectProps) => {
    return (
        <ActionButton
            aria-label={`Selected project ${name}`}
            isQuiet
            staticColor='white'
            UNSAFE_style={{
                paddingInline: 'var(--spectrum-global-dimension-size-200)',
                paddingBlock: 'var(--spectrum-global-dimension-size-200)',
            }}
        >
            <View margin={'size-50'}>{name}</View>
            <View margin='size-50'>
                <PhotoPlaceholder name={name} email='' height={'size-400'} width={'size-400'} />
            </View>
        </ActionButton>
    );
};

export const ProjectsListPanel = () => {
    const { project_id } = useProjectId();
    const { data: projects } = $api.useSuspenseQuery('get', '/api/projects');
    const { data: project } = $api.useSuspenseQuery('get', '/api/projects/{id}', {
        params: {
            path: { id: project_id },
        },
    });
    const otherProjects = projects.filter(({ id }) => id !== project_id);

    const selectedProjectName = project.name;

    return (
        <DialogTrigger type='popover' hideArrow>
            <SelectedProjectButton name={selectedProjectName} />

            <Dialog width={'size-4600'} UNSAFE_className={styles.dialog}>
                <Header>
                    <Flex direction={'column'} justifyContent={'center'} width={'100%'} alignItems={'center'}>
                        <PhotoPlaceholder
                            name={selectedProjectName}
                            email=''
                            height={'size-1000'}
                            width={'size-1000'}
                        />
                        <Heading level={2} marginBottom={0}>
                            {selectedProjectName}
                        </Heading>
                    </Flex>
                </Header>
                <Content UNSAFE_className={styles.panelContent}>
                    {otherProjects.length > 0 && (
                        <>
                            <Divider size={'S'} marginY={'size-200'} />
                            <ProjectsList projects={otherProjects} />
                        </>
                    )}
                    <Divider size={'S'} marginTop={'size-200'} />

                    <Link
                        href={paths.project.cameras.index({ project_id })}
                        UNSAFE_style={{ color: 'white', textDecoration: 'none' }}
                        UNSAFE_className={styles.openApiLink}
                    >
                        <Flex alignItems={'center'} gap='size-100'>
                            <ChevronRightSmallLight fill='white' />
                            Cameras
                        </Flex>
                    </Link>
                    <Link
                        href={paths.openapi({})}
                        UNSAFE_style={{ color: 'white', textDecoration: 'none' }}
                        UNSAFE_className={styles.openApiLink}
                    >
                        <Flex alignItems={'center'} gap='size-100'>
                            <ChevronRightSmallLight fill='white' />
                            OpenAPI Spec
                        </Flex>
                    </Link>

                    <Divider size={'S'} marginBottom={'size-200'} />
                </Content>

                <ButtonGroup UNSAFE_className={styles.panelButtons}>
                    <Link
                        href={paths.projects.new({})}
                        isQuiet
                        width={'100%'}
                        marginStart={'size-100'}
                        marginEnd={'size-350'}
                        UNSAFE_className={styles.addProjectButton}
                    >
                        <Icon>
                            <AddCircle />
                        </Icon>
                        <Text marginX='size-50'>Add project</Text>
                    </Link>
                </ButtonGroup>
            </Dialog>
        </DialogTrigger>
    );
};
