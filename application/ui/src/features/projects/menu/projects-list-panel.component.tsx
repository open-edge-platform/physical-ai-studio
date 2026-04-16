import { useState } from 'react';

import {
    ActionButton,
    ActionMenu,
    ButtonGroup,
    Content,
    Dialog,
    DialogContainer,
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
} from '@geti-ui/ui';
import { ChevronRightSmallLight, LogsIcon } from '@geti-ui/ui/icons';

import { $api } from '../../../api/client';
import { paths } from '../../../router';
import { LogsDialog } from '../../logs/logs-dialog';
import { NewProjectLink } from '../list/new-project-link.component';
import { useProjectId } from './../use-project';

import styles from './projects-list.module.scss';

interface Project {
    name: string;
    id?: string;
}
interface ProjectListProps {
    menuWidth?: string;
    projects: Project[];
}

export const ProjectsList = ({ menuWidth = '100%', projects }: ProjectListProps) => {
    const deleteMutation = $api.useMutation('delete', '/api/projects/{project_id}', {
        meta: {
            invalidates: [['get', '/api/projects']],
        },
    });

    return (
        <Menu UNSAFE_className={styles.projectMenu} width={menuWidth} items={projects}>
            {(item) => (
                <Item key={item.id} textValue={item.name} href={paths.project.robots.index({ project_id: item.id! })}>
                    <Text>
                        <Flex justifyContent='space-between' alignItems='center' marginX={'size-200'}>
                            {item.name}
                            <ActionMenu
                                isQuiet
                                onAction={(key) => {
                                    if (key === 'delete' && item.id) {
                                        deleteMutation.mutate({
                                            params: { path: { project_id: item.id } },
                                        });
                                    }
                                }}
                            >
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
                <PhotoPlaceholder name={name} indicator={name} height={'size-400'} width={'size-400'} />
            </View>
        </ActionButton>
    );
};

export const ProjectsListPanel = () => {
    const { project_id } = useProjectId();
    const { data: projects } = $api.useSuspenseQuery('get', '/api/projects');
    const { data: project } = $api.useSuspenseQuery('get', '/api/projects/{project_id}', {
        params: {
            path: { project_id },
        },
    });
    const otherProjects = projects.filter(({ id }) => id !== project_id);

    const [logsIsOpen, setLogsIsOpen] = useState(false);

    const selectedProjectName = project.name;

    return (
        <>
            <DialogTrigger type='popover' hideArrow>
                <SelectedProjectButton name={selectedProjectName} />

                {(close) => {
                    const handleOpenLogs = () => {
                        close();
                        setLogsIsOpen(true);
                    };
                    return (
                        <Dialog width={'size-4600'} UNSAFE_className={styles.dialog} onDismiss={close}>
                            <Header>
                                <Flex
                                    direction={'column'}
                                    justifyContent={'center'}
                                    width={'100%'}
                                    alignItems={'center'}
                                >
                                    <PhotoPlaceholder
                                        name={selectedProjectName}
                                        indicator={selectedProjectName}
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
                                    href={paths.openapi({})}
                                    UNSAFE_style={{ color: 'white', textDecoration: 'none' }}
                                    UNSAFE_className={styles.openApiLink}
                                >
                                    <Flex alignItems={'center'} gap='size-100'>
                                        <ChevronRightSmallLight fill='white' />
                                        OpenAPI Spec
                                    </Flex>
                                </Link>

                                <Divider size={'S'} />

                                <ActionButton
                                    isQuiet
                                    onPress={handleOpenLogs}
                                    UNSAFE_className={styles.openApiLink}
                                    width='100%'
                                    height='size-700'
                                >
                                    <Flex gap='size-50' marginEnd='size-100' width='100%'>
                                        <Icon size='S'>
                                            <LogsIcon />
                                        </Icon>
                                        Logs
                                    </Flex>
                                </ActionButton>

                                <Divider size={'S'} marginBottom={'size-200'} />
                            </Content>

                            <ButtonGroup UNSAFE_className={styles.panelButtons}>
                                <NewProjectLink className={styles.addProjectButton} />
                            </ButtonGroup>
                        </Dialog>
                    );
                }}
            </DialogTrigger>
            <DialogContainer onDismiss={() => setLogsIsOpen(false)} type='fullscreen'>
                {logsIsOpen && <LogsDialog close={() => setLogsIsOpen(false)} />}
            </DialogContainer>
        </>
    );
};
