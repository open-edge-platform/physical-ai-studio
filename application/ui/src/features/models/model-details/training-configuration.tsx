import { useState } from 'react';

import { Flex, Heading, Switch, View } from '@geti-ui/ui';

import type { SchemaModelDetailResponse } from '../../../api/openapi-spec';
import { Box } from '../shared/box';
import { DetailRow } from './detail-row';

const SKIP_HPARAMS_KEYS = new Set(['dataset_stats']);

export const TrainingParameters = ({ summary }: { summary: SchemaModelDetailResponse['training_summary'] }) => {
    if (summary === undefined || summary === null) {
        return null;
    }

    return (
        <View>
            <Heading>Training parameters</Heading>
            <Flex direction='column' gap='size-75'>
                {summary.max_epochs !== null && summary.max_epochs !== undefined ? (
                    <DetailRow name='Max epochs' value={summary.max_epochs} />
                ) : (
                    <DetailRow name='Max steps' value={summary.max_steps} />
                )}
                <DetailRow name='Batch size' value={summary.auto_scale_batch_size ? 'Auto' : summary.batch_size} />
                <DetailRow name='Workers' value={summary.num_workers ?? '—'} />
                {summary.precision && <DetailRow name='Precision' value={summary.precision} />}
                {summary.compile_model !== null && summary.compile_model !== undefined && (
                    <DetailRow name='Compiled' value={summary.compile_model ? 'Yes' : 'No'} />
                )}
                {summary.val_split !== null && summary.val_split !== undefined && summary.val_split > 0 && (
                    <DetailRow name='Validation split' value={summary.val_split} />
                )}
                {summary.device_type && <DetailRow name='Device' value={summary.device_type} />}
            </Flex>
        </View>
    );
};

export const HyperParameters = ({ hparams }: { hparams: SchemaModelDetailResponse['training_summary'] }) => {
    const [showJSON, setShowJSON] = useState(false);

    if (hparams === null) {
        return null;
    }

    return (
        <View>
            <View marginBottom={'size-100'}>
                <Flex direction='row' justifyContent={'space-between'}>
                    <Heading marginTop='size-200'>Hyper parameters</Heading>
                    <Switch isEmphasized isSelected={showJSON} onChange={setShowJSON}>
                        Show JSON
                    </Switch>
                </Flex>
            </View>
            <View maxHeight='60vh' overflow='auto'>
                {showJSON ? (
                    <View backgroundColor={'gray-100'} borderWidth='thin' borderColor='gray-200' paddingX='size-100'>
                        <pre>{JSON.stringify(hparams, null, 4)}</pre>
                    </View>
                ) : (
                    <Flex direction='column' gap='size-75'>
                        {hparams &&
                            Object.entries(hparams)
                                .filter(([key]) => !SKIP_HPARAMS_KEYS.has(key))
                                .map(([key, value]) => <DetailRow key={key} name={key} value={value} />)}
                    </Flex>
                )}
            </View>
        </View>
    );
};

export const TrainingConfiguration = ({
    summary,
    hparams,
}: {
    summary: SchemaModelDetailResponse['training_summary'];
    hparams: SchemaModelDetailResponse['training_summary'];
}) => {
    return (
        <View gridArea='training'>
            <Box
                title='Training configuration'
                content={
                    <Flex direction='column' gap='size-50'>
                        <TrainingParameters summary={summary} />

                        <HyperParameters hparams={hparams} />
                    </Flex>
                }
            />
        </View>
    );
};
