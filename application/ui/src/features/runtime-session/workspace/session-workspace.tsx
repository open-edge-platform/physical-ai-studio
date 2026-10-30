import { ReactNode, Suspense, useState } from 'react';

import { Button, Divider, Flex, Grid, Item, Loading, minmax, Picker, Text, View } from '@geti-ui/ui';

import { SessionPartName, useRuntimeSession } from '../runtime-session-provider';
import { SessionViewer } from '../session-viewer';
import { InferenceMode } from './inference-mode';
import { PartControlsProvider, usePartControls } from './part-controls';
import { PartsPanel } from './parts-panel';
import { RecordMode } from './record-mode';
import { ReplayMode } from './replay-mode';

/**
 * One way of working with the session: the parts it needs, and its side bar panel, which
 * drives the follower with the mode's own control. Modes are combined deliberately, as a
 * mode of their own (human in the loop), not by showing several panels at once.
 */
interface WorkspaceMode {
    key: ModeKey;
    label: string;
    needs: SessionPartName[];
    panel: ReactNode;
}

export type ModeKey = 'record' | 'replay' | 'inference';

const MODES: WorkspaceMode[] = [
    { key: 'record', label: 'Record', needs: ['environment', 'dataset'], panel: <RecordMode /> },
    { key: 'replay', label: 'Replay', needs: ['environment', 'dataset'], panel: <ReplayMode /> },
    { key: 'inference', label: 'Inference', needs: ['environment', 'model'], panel: <InferenceMode /> },
];

const ARTICLE: Record<SessionPartName, string> = {
    environment: 'an environment',
    dataset: 'a dataset',
    model: 'a model',
};

/** What a mode needs that the session has not loaded, each with a way to load it. */
const MissingParts = ({ needs }: { needs: SessionPartName[] }) => {
    const { state } = useRuntimeSession();
    const { pendingPart, act } = usePartControls();
    const missing = needs.filter((part) => state?.[part] == null);

    if (missing.length === 0) {
        return null;
    }
    return (
        <View backgroundColor='gray-200' padding='size-150' borderRadius='regular'>
            <Flex direction='column' gap='size-100'>
                {missing.map((part) => (
                    <Flex key={part} alignItems='center' justifyContent='space-between' gap='size-100'>
                        <Text>Needs {ARTICLE[part]}.</Text>
                        <Button
                            variant='secondary'
                            isPending={pendingPart === part}
                            // The dataset and the model load onto an environment.
                            isDisabled={part !== 'environment' && state?.environment == null}
                            onPress={() => act(part, 'swap')}
                        >
                            Load {part}…
                        </Button>
                    </Flex>
                ))}
            </Flex>
        </View>
    );
};

/**
 * The session's robot and cameras in a dock, with a side bar: what is loaded, the mode, and the
 * selected mode's panel. Every mode is always there and says what it misses; entry points
 * (record, run model) only choose what is loaded first and which mode is selected.
 */
export const SessionWorkspace = ({ initialMode, cancelHref }: { initialMode: ModeKey; cancelHref: string }) => {
    const { send } = useRuntimeSession();
    const [modeKey, setModeKey] = useState(initialMode);
    const mode = MODES.find((candidate) => candidate.key === modeKey) ?? MODES[0];

    const selectMode = (key: ModeKey) => {
        if (key === modeKey) {
            return;
        }
        // Each mode drives the follower its own way; hold it while switching.
        void send({ command: 'set_control', control: null });
        setModeKey(key);
    };

    return (
        <PartControlsProvider>
            <Grid
                columns={[minmax(0, '1fr'), 'size-4600']}
                gap='size-200'
                height='100%'
                minHeight={0}
                UNSAFE_style={{ gridTemplateRows: minmax(0, '1fr') }}
            >
                <SessionViewer cancelHref={cancelHref} />
                <View backgroundColor='gray-100' padding='size-200' overflow='auto' minHeight={0}>
                    <Flex direction='column' gap='size-200'>
                        <Suspense fallback={<Loading mode='inline' />}>
                            <PartsPanel />
                        </Suspense>
                        <Divider size='S' />
                        <Picker
                            label='Mode'
                            width='100%'
                            items={MODES}
                            selectedKey={mode.key}
                            onSelectionChange={(key) => key !== null && selectMode(key as ModeKey)}
                        >
                            {(candidate) => <Item key={candidate.key}>{candidate.label}</Item>}
                        </Picker>
                        <MissingParts needs={mode.needs} />
                        <Suspense fallback={<Loading mode='inline' />}>{mode.panel}</Suspense>
                    </Flex>
                </View>
            </Grid>
        </PartControlsProvider>
    );
};
