// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, Item, Key, Menu, MenuTrigger, type Selection } from '@geti-ui/ui';

import { ReactComponent as CameraIcon } from '../../../assets/camera.svg';
import { type DatasetCamera } from '../use-dataset-cameras';

interface ThumbnailCameraMenuProps {
    cameras: DatasetCamera[];
    selectedKey: string | undefined;
    onChange: (cameraKey: string) => void;
}

export const ThumbnailCameraMenu = ({ cameras, selectedKey, onChange }: ThumbnailCameraMenuProps) => {
    if (cameras.length < 2) {
        return null;
    }

    const selectedKeys: Iterable<Key> = selectedKey === undefined ? [] : [selectedKey];

    const handleSelectionChange = (keys: Selection) => {
        if (keys === 'all') {
            return;
        }

        const [cameraKey] = keys;

        if (cameraKey !== undefined) {
            onChange(String(cameraKey));
        }
    };

    return (
        <MenuTrigger>
            <ActionButton isQuiet aria-label='Thumbnail camera'>
                <CameraIcon />
            </ActionButton>
            <Menu
                items={cameras}
                selectionMode='single'
                disallowEmptySelection
                selectedKeys={selectedKeys}
                onSelectionChange={handleSelectionChange}
            >
                {(camera) => <Item key={camera.key}>{camera.name}</Item>}
            </Menu>
        </MenuTrigger>
    );
};
