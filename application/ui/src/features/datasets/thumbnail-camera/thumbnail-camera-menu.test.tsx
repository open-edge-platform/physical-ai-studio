// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { rs } from '@rstest/core';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { render } from '../../../test-utils/render';
import { ThumbnailCameraMenu } from './thumbnail-camera-menu';

const TOP = 'observation.images.top';
const WRIST = 'observation.images.wrist';

const cameras = [
    { key: TOP, name: 'top' },
    { key: WRIST, name: 'wrist' },
];

const renderThumbnailCameraMenu = (selectedKey: string | undefined) => {
    const onChange = rs.fn();
    const user = userEvent.setup();
    render(<ThumbnailCameraMenu cameras={cameras} selectedKey={selectedKey} onChange={onChange} />);

    return { user, onChange };
};

describe('ThumbnailCameraMenu', () => {
    it('renders nothing when there are fewer than two cameras', () => {
        const { unmount } = render(<ThumbnailCameraMenu cameras={[]} selectedKey={undefined} onChange={rs.fn()} />);
        expect(screen.queryByRole('button', { name: 'Thumbnail camera' })).not.toBeInTheDocument();
        unmount();

        render(<ThumbnailCameraMenu cameras={[{ key: TOP, name: 'top' }]} selectedKey={TOP} onChange={rs.fn()} />);
        expect(screen.queryByRole('button', { name: 'Thumbnail camera' })).not.toBeInTheDocument();
    });

    it('lists cameras by short name, checks the selected one and reports the full key on select', async () => {
        const { user, onChange } = renderThumbnailCameraMenu(TOP);

        await user.click(await screen.findByRole('button', { name: 'Thumbnail camera' }));

        const items = await screen.findAllByRole('menuitemradio');

        expect(items.map((item) => item.textContent)).toEqual(['top', 'wrist']);
        expect(screen.getByRole('menuitemradio', { name: 'top' })).toHaveAttribute('aria-checked', 'true');
        expect(screen.getByRole('menuitemradio', { name: 'wrist' })).toHaveAttribute('aria-checked', 'false');

        await user.click(screen.getByRole('menuitemradio', { name: 'wrist' }));

        expect(onChange).toHaveBeenCalledTimes(1);
        expect(onChange).toHaveBeenCalledWith(WRIST);
    });
});
