// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex, Heading, Link, Text, toast, View } from '@geti-ui/ui';

import { getApiErrorMessage } from '../../api/errors';
import { useFooterSimulations } from './use-footer-simulations';
import { useStopSimulation } from './use-simulation';

import classes from './simulation.module.css';

/** The Simulations section of the footer's runtime popover: Stop for Studio's, Open for others. */
export const SimulationsSection = () => {
    const simulations = useFooterSimulations();
    const stopMutation = useStopSimulation();

    if (simulations.length === 0) {
        return null;
    }

    return (
        <View marginTop='size-200'>
            <Heading level={4} margin={0} marginBottom='size-100'>
                Simulations
            </Heading>
            <Flex direction='column' gap='size-100'>
                {simulations.map((simulation) => {
                    return (
                        <Flex
                            key={simulation.key}
                            alignItems='center'
                            gap='size-150'
                            UNSAFE_className={classes.footerRow}
                        >
                            <Flex direction='column' flex={1} minWidth={0}>
                                <Text>
                                    {simulation.label}
                                    {simulation.scene !== null ? ` · ${simulation.scene}` : ''}
                                </Text>
                                <Text UNSAFE_className={classes.detail}>
                                    {simulation.startedByStudio ? 'Started by Studio' : 'Started outside Studio (CLI)'}
                                </Text>
                            </Flex>
                            {simulation.startedByStudio ? (
                                <Button
                                    variant='secondary'
                                    isPending={stopMutation.isPending}
                                    onPress={() =>
                                        stopMutation.mutate(
                                            {},
                                            {
                                                onError: (error) =>
                                                    toast.negative(getApiErrorMessage(error) ?? 'Failed to stop.'),
                                            }
                                        )
                                    }
                                >
                                    Stop
                                </Button>
                            ) : simulation.viewerUrl !== null ? (
                                <Link href={simulation.viewerUrl} target='_blank' rel='noreferrer' isQuiet>
                                    Open ↗
                                </Link>
                            ) : null}
                        </Flex>
                    );
                })}
            </Flex>
        </View>
    );
};
