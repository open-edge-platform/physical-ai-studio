import {
    ActionButton,
    Button,
    Content,
    Dialog,
    DialogTrigger,
    Divider,
    Flex,
    Heading,
    StatusLight,
    Text,
    View,
} from '@geti-ui/ui';
import useWebSocket from 'react-use-websocket';

import { $api, fetchClient } from '../../api/client';
import { SchemaRuntimeSessionInfo, SchemaRuntimeSessionsMessage } from '../../api/openapi-spec';

const SESSIONS_WS_URL = fetchClient.PATH('/api/runtime/sessions/ws');

/** The live runtime sessions, pushed by the backend on every change. One shared socket for all callers. */
export const useRuntimeSessions = () => {
    const { lastJsonMessage } = useWebSocket<SchemaRuntimeSessionsMessage | null>(SESSIONS_WS_URL, {
        share: true,
        shouldReconnect: () => true,
        reconnectInterval: 3000,
    });
    return { data: lastJsonMessage?.sessions };
};

/** The session holding a robot, if any. */
export const sessionForRobot = (sessions: SchemaRuntimeSessionInfo[] | undefined, robotId: string) =>
    sessions?.find((session) => session.robot_ids.includes(robotId));

const SessionRow = ({ session }: { session: SchemaRuntimeSessionInfo }) => {
    // No invalidation: the sessions socket pushes the new list.
    const stop = $api.useMutation('post', '/api/runtime/sessions/{session_id}/stop');

    return (
        <Flex alignItems='center' justifyContent='space-between' gap='size-200'>
            <Flex direction='column'>
                <Text>{session.environment_name ?? 'Nothing loaded'}</Text>
                <Text UNSAFE_style={{ fontSize: 'var(--spectrum-global-dimension-font-size-75)' }}>
                    {session.project_name} · since {new Date(session.started_at).toLocaleTimeString()}
                </Text>
            </Flex>
            <Button
                variant='negative'
                isPending={stop.isPending}
                onPress={() => stop.mutate({ params: { path: { session_id: session.session_id } } })}
            >
                Stop
            </Button>
        </Flex>
    );
};

/** Footer badge for the live runtime sessions, with a list to stop them from. Hidden when there are none. */
export const RuntimeSessionStatus = () => {
    const { data: sessions } = useRuntimeSessions();

    if (!sessions?.length) {
        return null;
    }

    return (
        <DialogTrigger type='popover'>
            <ActionButton isQuiet>
                <StatusLight variant='positive'>
                    {sessions.length === 1 ? '1 runtime session' : `${sessions.length} runtime sessions`}
                </StatusLight>
            </ActionButton>
            <Dialog>
                <Heading>Runtime sessions</Heading>
                <Divider />
                <Content>
                    <View>
                        <Flex direction='column' gap='size-200'>
                            {sessions.map((session) => (
                                <SessionRow key={session.session_id} session={session} />
                            ))}
                        </Flex>
                    </View>
                </Content>
            </Dialog>
        </DialogTrigger>
    );
};
