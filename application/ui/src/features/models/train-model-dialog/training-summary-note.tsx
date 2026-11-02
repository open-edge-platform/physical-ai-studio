import { Flex, Text, View } from '@geti-ui/ui';

import classes from './train-model-dialog.module.css';

interface TrainingSummaryNoteProps {
    device: string;
    dataset: string;
    policy: string;
}

const SummaryItem = ({ label, value }: { label: string; value: string }) => (
    <Flex direction='row' gap='size-75' alignItems='baseline'>
        <Text UNSAFE_className={classes.summaryLabel}>{label}</Text>
        <Text UNSAFE_className={classes.summaryValue}>{value}</Text>
    </Flex>
);

/**
 * Recap of what was picked on the setup step.
 *
 * The later steps configure a run whose device, dataset and policy are already
 * settled, and those three decide what the rest of the wizard even means — an
 * empty camera slot is fine for one policy and wrong for another — so they stay
 * in view instead of being a step away.
 */
export const TrainingSummaryNote = ({ device, dataset, policy }: TrainingSummaryNoteProps) => (
    <View backgroundColor='gray-75' borderRadius='regular' paddingX='size-150' paddingY='size-100'>
        <Flex direction='column' gap='size-300' wrap alignItems='baseline'>
            <SummaryItem label='Device' value={device} />
            <SummaryItem label='Dataset' value={dataset} />
            <SummaryItem label='Policy' value={policy} />
        </Flex>
    </View>
);
