import { useEffect, useId, useRef, useState } from 'react';

import {
    ActionButton,
    Button,
    ButtonGroup,
    Content,
    Dialog,
    DialogTrigger,
    Divider,
    Flex,
    Header,
    Heading,
    Icon,
    Loading,
    StatusLight,
    Text,
    ToastQueue,
    View,
} from '@geti-ui/ui';
import { Close, Pin } from '@geti-ui/ui/icons';

import { getApiErrorMessage } from '../../api/errors';
import { SchemaRuntimeSessionInfo } from '../../api/openapi-spec';
import { Table, TableColumn } from '../../components/table/table';
import {
    idleSecondsRemaining,
    sessionActivity,
    sessionLabel,
    sessionStatusVariant,
    uptimeLabel,
    useRuntimeSessionCount,
    useRuntimeSessions,
    useStopRuntimeSession,
} from './use-runtime-sessions';

import classes from './runtime-sessions.module.css';

const COLUMNS: TableColumn[] = [
    { width: 'max-content' },
    { width: '2fr', header: 'Robot' },
    { width: '1fr', header: 'Status' },
    { width: '1fr', header: 'Uptime' },
    { width: 'auto', align: 'end' },
];

const Field = ({ label, children }: { label: string; children: string }) => (
    <>
        <dt>{label}</dt>
        <dd>{children}</dd>
    </>
);

const SessionDetail = ({ session, now }: { session: SchemaRuntimeSessionInfo; now: number }) => {
    const idleSeconds = idleSecondsRemaining(session, now);
    const cameras = session.camera_keys ?? [];
    const activity = session.activity;
    const loadedLabel = (loaded: boolean | null | undefined) => (loaded ? 'Loaded' : 'Not loaded');

    return (
        <div className={classes.detail}>
            {idleSeconds !== undefined && (
                <Text UNSAFE_className={classes.abandoned}>
                    Nobody is watching this session. It shuts down in {idleSeconds}s.
                </Text>
            )}
            {session.error && (
                <Text UNSAFE_className={classes.errorMessage}>
                    {session.error.message} ({session.error.error_code})
                </Text>
            )}

            {(activity || session.leader_name || cameras.length > 0) && (
                <dl className={classes.fields}>
                    {activity?.task ? <Field label='Task'>{activity.task}</Field> : null}
                    {activity ? <Field label='Dataset'>{loadedLabel(activity.dataset_loaded)}</Field> : null}
                    {activity ? <Field label='Model'>{loadedLabel(activity.model_loaded)}</Field> : null}
                    {activity?.episodes_recorded != null && activity.episodes_recorded > 0 ? (
                        <Field label='Episodes recorded'>{String(activity.episodes_recorded)}</Field>
                    ) : null}
                    {session.leader_name ? <Field label='Leader robot'>{session.leader_name}</Field> : null}
                    {cameras.length > 0 ? <Field label='Cameras'>{cameras.join(', ')}</Field> : null}
                </dl>
            )}

            <dl className={`${classes.fields} ${classes.diagnostics}`}>
                {session.pid !== null && session.pid !== undefined ? (
                    <Field label='Process ID'>{String(session.pid)}</Field>
                ) : null}
                <Field label='Session ID'>{session.session_name}</Field>
            </dl>
        </div>
    );
};

/**
 * Inline confirmation attached to a row.
 *
 * Deliberately a ``group`` and not an ``alertdialog``: that role promises a
 * modal that traps focus and swallows Escape, and this bar is neither. It does
 * take focus on open and hand it back on cancel -- see ``SessionRow`` -- because
 * the button that opened it is disabled the moment it appears.
 */
const StopConfirm = ({
    label,
    isPending,
    onCancel,
    onConfirm,
}: {
    label: string;
    isPending: boolean;
    onCancel: () => void;
    onConfirm: () => void;
}) => {
    const headingId = useId();
    const descId = useId();
    const cancelId = useId();

    // Explicit rather than autoFocus: this is focus management in response to a
    // press, not focus stolen on load, and jsx-a11y rightly forbids the latter.
    useEffect(() => {
        document.getElementById(cancelId)?.focus();
    }, [cancelId]);

    return (
        <div
            role='group'
            aria-labelledby={headingId}
            aria-describedby={descId}
            className={classes.confirm}
            onClick={(event) => event.stopPropagation()}
            onKeyDown={(event) => {
                if (event.key === 'Escape') {
                    // Stopping a robot mid-task is worth an easy way out, and
                    // Escape must not reach the popover and close the whole list.
                    event.stopPropagation();
                    onCancel();
                }
            }}
        >
            <div>
                <Heading id={headingId} level={4} margin={0}>
                    {`Stop session for '${label}'?`}
                </Heading>
                <Text id={descId} UNSAFE_className={classes.confirmCopy}>
                    The session will be terminated.
                </Text>
            </div>
            <ButtonGroup>
                {/* Focus lands on the safe option, never on the destructive one. */}
                <Button id={cancelId} variant='secondary' onPress={onCancel}>
                    Cancel
                </Button>
                <Button variant='negative' onPress={onConfirm} isDisabled={isPending}>
                    Stop session
                </Button>
            </ButtonGroup>
        </div>
    );
};

const SessionRow = ({
    session,
    now,
    isConfirming,
    isStopPending,
    isExpanded,
    onExpandedChange,
    onStop,
    onCancelStop,
    onConfirmStop,
}: {
    session: SchemaRuntimeSessionInfo;
    now: number;
    isConfirming: boolean;
    isStopPending: boolean;
    isExpanded: boolean;
    onExpandedChange: (isExpanded: boolean) => void;
    onStop: () => void;
    onCancelStop: () => void;
    onConfirmStop: () => void;
}) => {
    const label = sessionLabel(session);
    const stopButtonId = useId();
    const wasConfirming = useRef(false);

    // Cancelling has to put the caret back where it came from. The Stop button
    // stays mounted so there is something to return to -- unmounting it would
    // drop focus to the document body, and a keyboard user would have to tab in
    // from the top of the page to reach the list again.
    useEffect(() => {
        if (wasConfirming.current && !isConfirming) {
            document.getElementById(stopButtonId)?.focus();
        }
        wasConfirming.current = isConfirming;
    }, [isConfirming, stopButtonId]);

    return (
        <Table.ExpandableRow
            label={`Details for ${label}`}
            detail={<SessionDetail session={session} now={now} />}
            isExpanded={isExpanded}
            onExpandedChange={onExpandedChange}
            after={
                isConfirming ? (
                    <StopConfirm
                        label={label}
                        isPending={isStopPending}
                        onCancel={onCancelStop}
                        onConfirm={onConfirmStop}
                    />
                ) : undefined
            }
        >
            <Text>{label}</Text>
            <StatusLight variant={sessionStatusVariant(session)} UNSAFE_className={classes.status}>
                {sessionActivity(session)}
            </StatusLight>
            <Text>{uptimeLabel(session, now) ?? '—'}</Text>
            <div onClick={(event) => event.stopPropagation()}>
                <Button
                    id={stopButtonId}
                    variant='negative'
                    onPress={onStop}
                    isDisabled={isConfirming}
                    aria-label={`Stop session for ${label}`}
                >
                    Stop
                </Button>
            </div>
        </Table.ExpandableRow>
    );
};

export const RuntimeSessionsDialog = ({
    close,
    isPinned = false,
    onPinnedChange,
    expandedNames,
    onExpandedNamesChange,
}: {
    close: () => void;
    isPinned?: boolean;
    onPinnedChange?: (isPinned: boolean) => void;
    expandedNames?: ReadonlySet<string>;
    onExpandedNamesChange?: (expandedNames: ReadonlySet<string>) => void;
}) => {
    const { data: sessions, isLoading } = useRuntimeSessions();
    const [stopTarget, setStopTarget] = useState<SchemaRuntimeSessionInfo | undefined>();
    const stopMutation = useStopRuntimeSession();
    const [internalExpanded, setInternalExpanded] = useState<ReadonlySet<string>>(() => new Set());
    const expanded = expandedNames ?? internalExpanded;

    const setRowExpanded = (sessionName: string, isExpanded: boolean) => {
        const next = new Set(expanded);
        if (isExpanded) {
            next.add(sessionName);
        } else {
            next.delete(sessionName);
        }
        if (onExpandedNamesChange !== undefined) {
            onExpandedNamesChange(next);
        } else {
            setInternalExpanded(next);
        }
    };

    // Uptime and the idle countdown are derived from timestamps, so they need a
    // tick of their own -- the poll alone would make them jump in 2s steps.
    const [now, setNow] = useState(() => Date.now());
    useEffect(() => {
        const timer = setInterval(() => setNow(Date.now()), 1_000);
        return () => clearInterval(timer);
    }, []);

    // Owned here rather than inside the confirmation, which unmounts as soon as
    // its primary action fires -- react-query drops the callbacks of an
    // unmounted component, and a failed stop leaves a robot held, so it must not
    // vanish with the dialog.
    // Scoped to the session actually being stopped. Shared across rows it would
    // disable the confirm on a *different* session while this one is in flight --
    // and a stop can take seconds, since the backend waits out SIGTERM before it
    // escalates. Derived from the mutation rather than tracked separately so it
    // cannot drift out of step with it.
    const pendingSessionName = stopMutation.isPending ? stopMutation.variables?.params?.path?.session_name : undefined;

    const stopSession = (session: SchemaRuntimeSessionInfo) => {
        const label = sessionLabel(session);
        stopMutation.mutate(
            { params: { path: { session_name: session.session_name } } },
            {
                onError: (error) =>
                    ToastQueue.negative(
                        getApiErrorMessage(error) ?? `The session for '${label}' could not be stopped. Try again.`
                    ),
            }
        );
        setStopTarget(undefined);
    };

    return (
        <Dialog UNSAFE_className={classes.dialog}>
            <Heading>Runtime sessions</Heading>
            <Header>
                <ActionButton
                    isQuiet
                    aria-label={isPinned ? 'Unpin session list' : 'Pin session list'}
                    aria-pressed={isPinned}
                    onPress={() => onPinnedChange?.(!isPinned)}
                    UNSAFE_className={`${classes.pinButton} ${isPinned ? classes.pinButtonActive : ''}`}
                >
                    <Icon>
                        <Pin />
                    </Icon>
                </ActionButton>
                <ActionButton isQuiet aria-label='Close' onPress={close} UNSAFE_className={classes.closeButton}>
                    <Icon>
                        <Close />
                    </Icon>
                </ActionButton>
            </Header>
            <Divider />
            <Content>
                {isLoading ? (
                    <Loading />
                ) : sessions === undefined || sessions.length === 0 ? (
                    <View UNSAFE_className={classes.empty}>
                        <Text>No runtime sessions are running.</Text>
                    </View>
                ) : (
                    <Table columns={COLUMNS}>
                        {sessions.map((session) => (
                            <SessionRow
                                key={session.session_name}
                                session={session}
                                now={now}
                                isConfirming={stopTarget?.session_name === session.session_name}
                                isStopPending={pendingSessionName === session.session_name}
                                isExpanded={expanded.has(session.session_name)}
                                onExpandedChange={(isExpanded) => setRowExpanded(session.session_name, isExpanded)}
                                onStop={() => setStopTarget(session)}
                                onCancelStop={() => setStopTarget(undefined)}
                                onConfirmStop={() => stopSession(session)}
                            />
                        ))}
                    </Table>
                )}
            </Content>
        </Dialog>
    );
};

/**
 * Footer entry point for the runtime sessions running on this host.
 *
 * Renders nothing when none are, matching the job status beside it. Polls the
 * count rather than the list: this is mounted on every page, and the count is a
 * directory read while the list opens a transport session per runtime session.
 */
export const RuntimeSessionStatus = () => {
    const { data } = useRuntimeSessionCount();
    const count = data?.count ?? 0;
    const [isOpen, setIsOpen] = useState(false);
    const [isPinned, setIsPinned] = useState(false);
    const isPinnedRef = useRef(false);
    // Kept outside the overlay so a close/reopen can restore which rows were open.
    const [expandedNames, setExpandedNames] = useState<ReadonlySet<string>>(() => new Set());

    useEffect(() => {
        if (!isOpen) {
            return;
        }

        const onPointerDown = (event: PointerEvent) => {
            if (isPinnedRef.current) {
                return;
            }
            const target = event.target;
            if (!(target instanceof Element)) {
                return;
            }
            if (target.closest('[role="dialog"]') !== null || target.closest('[data-testid="popover"]') !== null) {
                return;
            }
            if (target.closest('[aria-label="Runtime sessions"]') !== null) {
                return;
            }
            setIsOpen(false);
        };

        document.addEventListener('pointerdown', onPointerDown, true);
        return () => document.removeEventListener('pointerdown', onPointerDown, true);
    }, [isOpen]);

    // Stay mounted while the panel is open, even at zero. Unmounting here takes
    // the popover with it, so stopping your last session used to yank the panel
    // away mid-interaction -- and a pinned one closing is the opposite of what
    // pinning asks for. It also left the empty state unreachable from here.
    if (count === 0 && !isOpen) {
        return null;
    }

    const setPinned = (pinned: boolean) => {
        isPinnedRef.current = pinned;
        setIsPinned(pinned);
        if (pinned) {
            setIsOpen(true);
        }
    };

    const dismiss = () => {
        isPinnedRef.current = false;
        setIsPinned(false);
        setIsOpen(false);
        setExpandedNames(new Set());
    };

    // Always non-modal. Toggling that flag (or focus containment) remounts
    // Spectrum's overlay, which is the flash you see on pin. Click-away is
    // handled above so an unpinned list still dismisses.
    const popover = {
        type: 'popover' as const,
        placement: 'top' as const,
        isOpen,
        isKeyboardDismissDisabled: isPinned,
        isNonModal: true,
        disableFocusManagement: true,
        shouldCloseOnInteractOutside: () => !isPinned,
    };

    return (
        <DialogTrigger
            {...popover}
            onOpenChange={(open) => {
                if (!open && isPinnedRef.current) {
                    setIsOpen(true);
                    return;
                }
                setIsOpen(open);
            }}
        >
            <ActionButton isQuiet aria-label='Runtime sessions'>
                <Flex alignItems='center' gap='size-50'>
                    <StatusLight variant='positive' marginEnd='size-0' />
                    <Text>{count === 1 ? '1 session' : `${count} sessions`}</Text>
                </Flex>
            </ActionButton>
            {() => (
                <RuntimeSessionsDialog
                    close={dismiss}
                    isPinned={isPinned}
                    onPinnedChange={setPinned}
                    expandedNames={expandedNames}
                    onExpandedNamesChange={setExpandedNames}
                />
            )}
        </DialogTrigger>
    );
};
