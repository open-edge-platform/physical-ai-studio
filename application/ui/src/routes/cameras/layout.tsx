import { Suspense, useState } from 'react';

import {
    ActionButton,
    Button,
    ButtonGroup,
    Content,
    Dialog,
    DialogContainer,
    Divider,
    Flex,
    Grid,
    Heading,
    Icon,
    Item,
    Loading,
    Menu,
    MenuTrigger,
    minmax,
    toast,
    View,
} from '@geti-ui/ui';
import { Add, MoreMenu } from '@geti-ui/ui/icons';
import { clsx } from 'clsx';
import { NavLink, Outlet, useParams } from 'react-router';

import { $api } from '../../api/client';
import { getApiErrorMessage, isRecordingLockedError, isResourceInUseError } from '../../api/errors';
import { SchemaProjectCamera } from '../../api/types';
import { fingerprintKey, formatFingerprint } from '../../features/cameras/fingerprint';
import { useProjectId } from '../../features/projects/use-project';
import { ConnectionStatus } from '../../features/robots/robots-list';
import { paths } from '../../router';
import { ReactComponent as CameraIcon } from './../../assets/camera.svg';

import classes from './../../features/robots/robots-list.module.css';

const CopyFingerprintDialog = ({
    fingerprint,
    onDismiss,
}: {
    fingerprint: SchemaProjectCamera['fingerprint'];
    onDismiss: () => void;
}) => {
    const json = JSON.stringify(fingerprint ?? null, null, 2);

    const handleCopy = async () => {
        try {
            await navigator.clipboard.writeText(json);
            toast.positive('Fingerprint copied to clipboard.');
        } catch {
            toast.negative('Failed to copy fingerprint to clipboard.');
        }
    };

    return (
        <DialogContainer onDismiss={onDismiss}>
            <Dialog>
                <Heading>Hardware fingerprint</Heading>
                <Divider />
                <Content>
                    <View
                        backgroundColor='gray-100'
                        borderRadius='regular'
                        padding='size-200'
                        overflow='auto'
                        maxHeight='size-4600'
                    >
                        <pre style={{ margin: 0, fontSize: '12px', whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
                            {json}
                        </pre>
                    </View>
                </Content>
                <ButtonGroup>
                    <Button variant='secondary' onPress={onDismiss}>
                        Close
                    </Button>
                    <Button variant='accent' onPress={handleCopy}>
                        Copy to clipboard
                    </Button>
                </ButtonGroup>
            </Dialog>
        </DialogContainer>
    );
};

const MenuActions = ({ camera }: { camera: SchemaProjectCamera }) => {
    const camera_id = camera.id ?? 'undefined';
    const { project_id } = useProjectId();
    const [isFingerprintDialogOpen, setIsFingerprintDialogOpen] = useState(false);
    const deleteCameraMutation = $api.useMutation('delete', '/api/projects/{project_id}/cameras/{camera_id}', {
        meta: {
            invalidates: [['get', '/api/projects/{project_id}/cameras', { params: { path: { project_id } } }]],
        },
    });

    return (
        <>
            <MenuTrigger>
                <ActionButton isQuiet>
                    <MoreMenu />
                </ActionButton>
                <Menu
                    selectionMode='single'
                    onAction={(action) => {
                        if (action === 'show-fingerprint') {
                            setIsFingerprintDialogOpen(true);
                            return;
                        }
                        if (action === 'delete') {
                            deleteCameraMutation.mutate(
                                { params: { path: { project_id, camera_id } } },
                                {
                                    onError: (error) => {
                                        if (isRecordingLockedError(error)) {
                                            toast.negative('Cannot delete camera while a recording session is active.');
                                            return;
                                        }
                                        if (isResourceInUseError(error)) {
                                            toast.info(
                                                getApiErrorMessage(error) ??
                                                    'This camera is in use and cannot be deleted.'
                                            );
                                            return;
                                        }
                                        toast.negative(getApiErrorMessage(error) ?? 'Failed to delete camera.');
                                    },
                                }
                            );
                        }
                    }}
                >
                    <Item href={paths.project.cameras.edit({ project_id, camera_id })}>Edit</Item>
                    <Item key='show-fingerprint'>Show fingerprint</Item>
                    <Item key='delete'>Delete</Item>
                </Menu>
            </MenuTrigger>

            {isFingerprintDialogOpen && (
                <CopyFingerprintDialog
                    fingerprint={camera.fingerprint}
                    onDismiss={() => setIsFingerprintDialogOpen(false)}
                />
            )}
        </>
    );
};

const CameraListItem = ({
    status,
    camera,
    isActive,
}: {
    status: 'connected' | 'disconnected';
    camera: SchemaProjectCamera;
    isActive: boolean;
}) => {
    return (
        <View
            padding='size-200'
            UNSAFE_className={clsx({
                [classes.robotListItem]: true,
                [classes.robotListItemActive]: isActive,
            })}
        >
            <Flex direction={'column'} justifyContent={'space-between'} gap={'size-50'}>
                <Grid
                    areas={['icon name status', 'parameters parameters menu']}
                    columns={['auto', '1fr']}
                    gap={'size-100'}
                >
                    <View gridArea={'icon'} padding='size-100'>
                        <CameraIcon style={{ width: '32px', height: '32px' }} />
                    </View>
                    <View gridArea='name'>
                        <Heading level={2} UNSAFE_style={isActive ? { color: 'var(--energy-blue)' } : {}}>
                            {camera.name}
                        </Heading>
                        <View UNSAFE_style={{ fontSize: '14px' }}>
                            {camera.payload.width} x {camera.payload.height} @ {camera.payload.fps}
                        </View>
                    </View>
                    <View gridArea='status'>
                        <ConnectionStatus status={status == 'connected' ? 'online' : 'offline'} />
                    </View>
                    <View gridArea='menu' alignSelf={'end'} justifySelf={'end'}>
                        <MenuActions camera={camera} />
                    </View>
                    <View gridArea='parameters'>
                        <ul
                            style={{
                                display: 'flex',
                                flexDirection: 'column',
                                gap: 'var(--spectrum-global-dimension-size-10)',
                                listStyleType: 'disc',
                                fontSize: '10px',
                            }}
                        >
                            <li style={{ marginLeft: 'var(--spectrum-global-dimension-size-200)' }}>
                                {camera.driver}: {camera.hardware_name}
                            </li>
                            <li style={{ marginLeft: 'var(--spectrum-global-dimension-size-200)' }}>
                                {formatFingerprint(camera.fingerprint)}
                            </li>
                        </ul>
                    </View>
                </Grid>
            </Flex>
        </View>
    );
};

export const CamerasList = () => {
    const { project_id = '' } = useParams<{ project_id: string }>();
    const { data: hardwareCameras } = $api.useSuspenseQuery('get', '/api/hardware/cameras', {
        params: { query: { all: true } },
    });
    const { data: projectCameras } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/cameras', {
        params: { path: { project_id } },
    });

    return (
        <Flex direction='column' gap='size-100'>
            {/* TODO:  */}
            <View isHidden>
                <Flex justifyContent={'space-between'} alignItems={'end'}>
                    <span>Step 2: setup cameras</span>
                    <Button>Next</Button>
                </Flex>
                <Divider size='S' marginY='size-200' />
            </View>
            <Button
                variant='secondary'
                href={paths.project.cameras.new({ project_id })}
                UNSAFE_className={classes.addNewRobotButton}
            >
                <Icon marginEnd='size-50'>
                    <Add />
                </Icon>
                Configure new camera
            </Button>

            <Flex direction='column' gap='size-100'>
                {projectCameras.map((camera) => {
                    const cameraFingerprint = fingerprintKey(camera.fingerprint);
                    const hardwareCamera = cameraFingerprint
                        ? hardwareCameras.find((hardware) => fingerprintKey(hardware.fingerprint) === cameraFingerprint)
                        : undefined;
                    const to = paths.project.cameras.show({ project_id, camera_id: camera.id ?? 'undefined' });

                    return (
                        <NavLink key={camera.id} to={to}>
                            {({ isActive }) => {
                                return (
                                    <CameraListItem
                                        camera={camera}
                                        isActive={isActive}
                                        status={hardwareCamera !== undefined ? 'connected' : 'disconnected'}
                                    />
                                );
                            }}
                        </NavLink>
                    );
                })}
            </Flex>
        </Flex>
    );
};

export const Layout = () => {
    return (
        <Grid
            areas={['camera controls']}
            columns={['size-6000', minmax(0, '1fr')]}
            rows={[minmax(0, '1fr')]}
            height={'100%'}
            minHeight={0}
        >
            <View gridArea='camera' backgroundColor={'gray-100'} padding='size-400'>
                <CamerasList />
            </View>
            <View
                gridArea='controls'
                backgroundColor={'gray-50'}
                height='100%'
                minHeight={0}
                minWidth={0}
                overflow='auto'
            >
                <Suspense
                    fallback={
                        <Grid width='100%' height='100%'>
                            <Loading mode='inline' />
                        </Grid>
                    }
                >
                    <Outlet />
                </Suspense>
            </View>
        </Grid>
    );
};
