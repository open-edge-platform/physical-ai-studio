// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { render } from '../../../test-utils/render';
import { MetricGraph } from './metric-graph';
import { MetricTooltipContent } from './metric-tooltip';
import type { MetricLine, MetricsEntry } from './types';

const data: MetricsEntry[] = [{ step: 65, epoch: 0, train_loss: null, val_loss: null, mean: 4, joint: 2 }];
const lines: MetricLine[] = [
    { name: 'Mean', color: 'blue', primary: true, getY: (row) => row.mean },
    { name: 'Joint', color: 'orange', getY: (row) => row.joint },
];
const graphProps = {
    title: 'Action error',
    yAxisLabel: 'Error',
    getX: (row: MetricsEntry) => row.step,
    getY: (row: MetricsEntry) => row.mean,
    lines,
};

const StreamingGraph = () => {
    const [rows, setRows] = useState(data);
    return (
        <>
            <button onClick={() => setRows([...rows, { ...data[0], step: 131 }])}>Append metrics</button>
            <MetricGraph {...graphProps} data={rows} />
        </>
    );
};

describe('metric series controls', () => {
    it('toggles series with keyboard and pointer and preserves choices when data streams in', async () => {
        const user = userEvent.setup();
        render(<StreamingGraph />);
        const mean = screen.getByRole('button', { name: 'Mean' });
        const joint = screen.getByRole('button', { name: 'Joint' });

        expect(mean).toHaveAttribute('aria-pressed', 'true');
        await user.click(joint);
        expect(joint).toHaveAttribute('aria-pressed', 'false');
        expect(mean).toHaveAttribute('aria-pressed', 'true');

        await user.click(screen.getByRole('button', { name: 'Append metrics' }));
        expect(joint).toHaveAttribute('aria-pressed', 'false');
        joint.focus();
        await user.keyboard(' ');
        expect(joint).toHaveAttribute('aria-pressed', 'true');
    });

    it('allows every series to be hidden and restored', async () => {
        const user = userEvent.setup();
        render(<MetricGraph {...graphProps} data={data} />);
        const region = screen.getByRole('region', { name: 'Action error graph' });
        for (const button of within(region).getAllByRole('button')) {
            await user.click(button);
            expect(button).toHaveAttribute('aria-pressed', 'false');
        }

        await user.click(within(region).getByRole('button', { name: 'Mean' }));
        expect(within(region).getByRole('button', { name: 'Mean' })).toHaveAttribute('aria-pressed', 'true');
    });

    it('highlights a line from its tooltip row on hover and keyboard focus', async () => {
        const highlighted: Array<string | undefined> = [];
        const user = userEvent.setup();
        render(
            <MetricTooltipContent
                title='Action error'
                active
                label={65}
                lines={lines}
                payload={[
                    { name: 'Mean', value: 4, graphicalItemId: 'mean' },
                    { name: 'Joint', value: 2, graphicalItemId: 'joint' },
                ]}
                onHighlight={(name) => highlighted.push(name)}
            />
        );

        await user.hover(screen.getByRole('button', { name: 'Highlight Joint' }));
        expect(highlighted.at(-1)).toBe('Joint');
        await user.unhover(screen.getByRole('button', { name: 'Highlight Joint' }));
        expect(highlighted.at(-1)).toBeUndefined();
        await user.tab();
        expect(highlighted.at(-1)).toBe('Mean');
        expect(screen.getByRole('tooltip')).toHaveTextContent('Step: 65');
    });

    it('excludes hidden tooltip rows without dropping missing values from visible series', () => {
        render(
            <MetricTooltipContent
                title='Action error'
                active
                label={65}
                lines={lines}
                payload={[
                    { name: 'Mean', value: undefined, graphicalItemId: 'mean' },
                    { name: 'Joint', value: 2, hide: true, graphicalItemId: 'joint' },
                ]}
                onHighlight={() => {}}
            />
        );

        expect(screen.getByRole('tooltip')).toHaveTextContent('Mean: Not available');
        expect(screen.queryByRole('button', { name: 'Highlight Joint' })).not.toBeInTheDocument();
    });

    it('labels synchronized values with the selected sample step rather than the source label', () => {
        render(
            <MetricTooltipContent
                title='Action error'
                active
                label={99}
                activeIndex='0'
                data={data}
                getX={graphProps.getX}
                lines={lines}
                payload={[{ name: 'Mean', value: 4, graphicalItemId: 'mean' }]}
                onHighlight={() => {}}
            />
        );

        expect(screen.getByRole('tooltip')).toHaveTextContent('Step: 65');
        expect(screen.getByRole('tooltip')).not.toHaveTextContent('Step: 99');
    });
});
