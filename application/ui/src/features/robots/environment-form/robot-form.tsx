import { useState } from 'react';

import {
    Accordion,
    ActionButton,
    Button,
    Disclosure,
    DisclosurePanel,
    DisclosureTitle,
    Flex,
    Heading,
    Icon,
    Item,
    Picker,
    Text,
    View,
    Well,
} from '@geti/ui';
import { Add, Close } from '@geti/ui/icons';

import { $api } from '../../../api/client';
import { useProjectId } from '../../../features/projects/use-project';
import { RobotConfiguration, useEnvironmentForm, useSetEnvironmentForm } from './provider';

import classes from './form.module.scss';

const RobotListItem = ({ robot, onRemove }: { robot: RobotConfiguration; onRemove: () => void }) => {
    const { project_id } = useProjectId();
    const robotsQuery = $api.useSuspenseQuery('get', '/api/projects/{project_id}/robots', {
        params: { path: { project_id } },
    });

    const followerRobot = robotsQuery.data.find(({ id }) => id === robot.robot_id);

    if (followerRobot === undefined) {
        return <li>{robot.robot_id} - unknown</li>;
    }

    const leaderRobot = robotsQuery.data.find(
        ({ id }) => robot.teleoperator.type === 'robot' && robot.teleoperator.robot_id === id
    );
    if (leaderRobot === undefined) {
        return <li>Unknown leader robot</li>;
    }

    return (
        <li>
            <View backgroundColor={'gray-50'} padding='size-200' borderColor='gray-200' borderWidth='thick'>
                <Flex justifyContent='space-between' alignItems={'center'}>
                    <Flex direction='column' gap='size-100'>
                        <Flex gap='size-200'>
                            <span>Robot</span>
                            <span>{followerRobot.name}</span>
                        </Flex>
                        <Flex gap='size-200'>
                            <span>Tele operator</span>
                            <span>{leaderRobot.name}</span>
                        </Flex>
                    </Flex>

                    <ActionButton onPress={onRemove} UNSAFE_className={classes.actionButton}>
                        <Icon>
                            <Close />
                        </Icon>
                    </ActionButton>
                </Flex>
            </View>
        </li>
    );
};

export const AddRobotForm = ({
    onAddRobot,

    onCancel,
}: {
    onAddRobot: (robot: RobotConfiguration) => void;

    onCancel?: () => void;
}) => {
    const [selectedRobotId, setSelectedRobotId] = useState<string | null>(null);
    const [selectedTeleoperatorRobotId, setSelectedTeleoperatorRobotId] = useState<string | null>(null);

    const { project_id } = useProjectId();
    const robotsQuery = $api.useSuspenseQuery('get', '/api/projects/{project_id}/robots', {
        params: { path: { project_id } },
    });
    const environment = useEnvironmentForm();

    const availableRobots = robotsQuery.data.filter((robot) => {
        return (
            environment.robots.some(({ robot_id, teleoperator }) => {
                if (robot_id === robot.id) {
                    return true;
                }

                if (teleoperator.type === 'robot' && teleoperator.robot_id === robot.id) {
                    return true;
                }

                return false;
            }) === false
        );
    });

    if (availableRobots.length === 0) {
        return <span>No available robots</span>;
    }

    return (
        <Flex direction='column' gap='size-100'>
            <Heading level={4}>Robot</Heading>

            <Picker
                label='Robot (Follower)'
                width='100%'
                selectedKey={selectedRobotId}
                onSelectionChange={(key) => {
                    if (key !== null && typeof key === 'string') {
                        setSelectedRobotId(key);
                    }
                }}
            >
                {availableRobots
                    .filter((robot) => robot.type === 'SO101_Follower' || robot.type === 'Trossen_WidowXAI_Follower')
                    .map((robot) => {
                        return (
                            <Item textValue={robot.name} key={robot.id}>
                                <Text>{robot.name}</Text>
                            </Item>
                        );
                    })}
            </Picker>

            <Accordion defaultExpandedKeys={['leader-robot']} marginY='size-200'>
                <Disclosure id='none'>
                    <DisclosureTitle>No teleoperator</DisclosureTitle>
                    <DisclosurePanel>No teleoperator</DisclosurePanel>
                </Disclosure>
                <Disclosure id='leader-robot'>
                    <DisclosureTitle>Leader robot</DisclosureTitle>
                    <DisclosurePanel>
                        <Picker
                            label='Robot (Leader)'
                            width='100%'
                            selectedKey={selectedTeleoperatorRobotId}
                            onSelectionChange={(key) => {
                                if (key !== null && typeof key === 'string') {
                                    setSelectedTeleoperatorRobotId(key);
                                }
                            }}
                        >
                            {availableRobots
                                .filter(
                                    (robot) => robot.type === 'SO101_Leader' || robot.type === 'Trossen_WidowXAI_Leader'
                                )
                                .map((robot) => {
                                    return (
                                        <Item textValue={robot.name} key={robot.id}>
                                            <Text>{robot.name}</Text>
                                        </Item>
                                    );
                                })}
                        </Picker>
                    </DisclosurePanel>
                </Disclosure>
            </Accordion>

            <Flex gap='size-100'>
                <Button
                    variant='secondary'
                    onPress={() => {
                        if (selectedRobotId && selectedTeleoperatorRobotId) {
                            onAddRobot({
                                robot_id: selectedRobotId,
                                teleoperator: { robot_id: selectedTeleoperatorRobotId, type: 'robot' },
                            });
                        }
                    }}
                >
                    Add
                </Button>

                {onCancel && (
                    <Button variant='secondary' onPress={onCancel}>
                        Cancel
                    </Button>
                )}
            </Flex>
        </Flex>
    );
};

export const RobotForm = () => {
    const environmentForm = useEnvironmentForm();
    const setEnvironmentForm = useSetEnvironmentForm();

    const hasNoRobots = environmentForm.robots.length === 0;
    const [isAdding, setIsAdding] = useState(hasNoRobots);

    return (
        <>
            <ul style={{ width: '100%' }}>
                <Flex direction='column' gap='size-100' width='100%'>
                    {environmentForm.robots.map((robot) => (
                        <RobotListItem
                            key={robot.robot_id}
                            robot={robot}
                            onRemove={() => {
                                setEnvironmentForm((oldForm) => {
                                    return {
                                        ...oldForm,
                                        robots: oldForm.robots.filter(({ robot_id }) => robot_id !== robot.robot_id),
                                    };
                                });
                            }}
                        />
                    ))}
                </Flex>
            </ul>

            {isAdding ? (
                <Well
                    width='100%'
                    UNSAFE_style={{
                        backgroundColor: 'var(--spectrum-global-color-gray-200)',
                    }}
                >
                    <AddRobotForm
                        onAddRobot={(robot) => {
                            setEnvironmentForm((oldForm) => {
                                return {
                                    ...oldForm,
                                    robots: [...oldForm.robots, robot],
                                };
                            });
                            setIsAdding(false);
                        }}
                        onCancel={
                            hasNoRobots
                                ? undefined
                                : () => {
                                      setIsAdding(false);
                                  }
                        }
                    />
                </Well>
            ) : environmentForm.robots.length === 0 ? (
                <Button
                    variant='secondary'
                    UNSAFE_className={classes.addNewButton}
                    width='100%'
                    onPress={() => {
                        setIsAdding(true);
                    }}
                >
                    <Icon marginEnd='size-50'>
                        <Add />
                    </Icon>
                    Robot
                </Button>
            ) : null}
        </>
    );
};
