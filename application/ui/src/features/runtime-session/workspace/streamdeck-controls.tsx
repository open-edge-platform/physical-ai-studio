import { useEffect, useRef } from 'react';

import { Button, Flex, Text } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { getApiErrorMessage } from '../../../api/errors';
import { SchemaModelControlConfig } from '../../../api/openapi-spec';

interface StreamDeckControlsProps {
    sessionId: string;
    task: string;
    canConnect: boolean;
    page?: 'record' | 'inference';
    inference?: SchemaModelControlConfig;
    onTaskChange?: (task: string) => void;
    onDictationBusyChange?: (busy: boolean) => void;
}

/** The helper runs on the USB host, whether Studio runs natively or in Docker. */
export const StreamDeckControls = ({
    sessionId,
    task,
    canConnect,
    page = 'record',
    inference,
    onTaskChange,
    onDictationBusyChange,
}: StreamDeckControlsProps) => {
    const { data: status, isError } = $api.useQuery('get', '/api/streamdeck', {}, { refetchInterval: 1000 });
    const start = $api.useMutation('post', '/api/streamdeck/start', {
        meta: { awaits: [['get', '/api/streamdeck']] },
    });
    const connect = $api.useMutation('put', '/api/streamdeck/sessions/{session_id}', {
        meta: { awaits: [['get', '/api/streamdeck']] },
    });
    const disconnect = $api.useMutation('delete', '/api/streamdeck/sessions/{session_id}', {
        meta: { awaits: [['get', '/api/streamdeck']] },
    });
    const { mutate: attach, isPending } = connect;
    const ownTarget = status?.target?.session_id === sessionId ? status.target : null;
    const connection = ownTarget !== null && (ownTarget.page ?? 'record') === page ? ownTarget : null;
    const attemptedTask = useRef('');
    const lastTranscript = useRef<string | null>(null);
    const busy =
        connection !== null && ['preparing', 'listening', 'transcribing'].includes(status?.dictation_status ?? 'idle');
    const settings = JSON.stringify([
        inference?.hz,
        inference?.execution?.mode,
        inference?.execution?.request_threshold,
        inference?.grace_s,
    ]);
    const remoteSettings = JSON.stringify([
        connection?.inference?.hz,
        connection?.inference?.execution?.mode,
        connection?.inference?.execution?.request_threshold,
        connection?.inference?.grace_s,
    ]);
    const synchronized = connection?.task === task && (page === 'record' || settings === remoteSettings);

    useEffect(() => {
        onDictationBusyChange?.(busy);
    }, [busy, onDictationBusyChange]);

    useEffect(() => {
        const transcript = connection?.dictation;
        if (transcript && onTaskChange && transcript.id !== lastTranscript.current) {
            lastTranscript.current = transcript.id;
            attemptedTask.current = '';
            onTaskChange(transcript.task);
        }
    }, [connection?.dictation, onTaskChange]);

    useEffect(() => {
        if (connection === null || synchronized) {
            attemptedTask.current = '';
            return;
        }
        if (isPending || disconnect.isPending || isError) return;
        const attempt = JSON.stringify([connection.connection_id, task, settings]);
        if (attemptedTask.current === attempt) return;
        const timer = window.setTimeout(() => {
            attemptedTask.current = attempt;
            attach({ params: { path: { session_id: sessionId } }, body: { task, page, inference } });
        }, 350);
        return () => window.clearTimeout(timer);
    }, [
        connection,
        synchronized,
        task,
        page,
        inference,
        settings,
        sessionId,
        attach,
        isPending,
        disconnect.isPending,
        isError,
    ]);

    const otherWorkspace = status?.target != null && ownTarget === null;
    const helperOnline = !isError && status?.helper_online;
    const error = start.error ?? connect.error ?? disconnect.error;
    const helperMessage = !status
        ? 'Checking Stream Deck helper…'
        : status.helper_starting
          ? 'Starting Stream Deck helper…'
          : status.can_start_native
            ? 'Start the helper on this Studio computer.'
            : 'Open the installed host launcher and approve the browser prompt. First use needs one-time setup:';
    const pageName = page === 'record' ? 'recording' : 'inference';
    const message = isError
        ? 'Stream Deck status unavailable.'
        : !helperOnline
          ? helperMessage
          : otherWorkspace
            ? 'Connected in another workspace. Disconnect there first.'
            : connection !== null
              ? status?.connected
                  ? `${status.device} connected${!synchronized ? ' — synchronizing task…' : '.'}`
                  : 'Connecting to the USB deck…'
              : ownTarget !== null
                ? `Switch the deck to this workspace’s ${pageName} page.`
                : `Helper ready. Connect to this workspace’s ${pageName} page.`;

    return (
        <Flex direction='column' gap='size-100'>
            <Text>
                <strong>Stream Deck</strong>
            </Text>
            <div role='status' aria-live='polite'>
                {connect.isError && connection !== null && !synchronized
                    ? `Task sync failed. Disconnect Stream Deck before ${pageName}.`
                    : busy
                      ? `Local microphone: ${status?.dictation_status}. Inference stays stopped.`
                      : message}
            </div>
            {!helperOnline && !isError && status && (
                <>
                    {!status.can_start_native && <code>uv run application/streamdeck/host_launcher.py --install</code>}
                    <Button
                        variant='secondary'
                        href={status.can_start_native ? undefined : 'physicalai-streamdeck://start'}
                        isPending={start.isPending || status.helper_starting}
                        onPress={() => {
                            if (status.can_start_native) {
                                start.mutate({ params: { header: { 'x-streamdeck-launch': '1' } } });
                            }
                        }}
                    >
                        Start Stream Deck helper
                    </Button>
                </>
            )}
            {(error != null || status?.error) && (
                <Text>{getApiErrorMessage(error) ?? status?.error ?? 'Stream Deck connection failed.'}</Text>
            )}
            <Button
                variant='secondary'
                isPending={isPending || disconnect.isPending}
                isDisabled={
                    connection === null &&
                    (!helperOnline || otherWorkspace || !canConnect || (page === 'record' && !task.trim()))
                }
                onPress={() => {
                    connect.reset();
                    disconnect.reset();
                    attemptedTask.current = '';
                    if (connection !== null) {
                        disconnect.mutate({ params: { path: { session_id: sessionId } } });
                    } else {
                        attach({ params: { path: { session_id: sessionId } }, body: { task, page, inference } });
                    }
                }}
            >
                {connection !== null
                    ? 'Disconnect Stream Deck'
                    : ownTarget !== null
                      ? `Use ${page === 'record' ? 'recording' : 'inference'} page`
                      : 'Connect Stream Deck'}
            </Button>
        </Flex>
    );
};
