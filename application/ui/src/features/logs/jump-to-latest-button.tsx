// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, Icon, Text } from '@geti-ui/ui';
import { ChevronDownLight } from '@geti-ui/ui/icons';

import styles from './log-viewer.module.css';

export const JumpToLatestButton = ({ isVisible, onPress }: { isVisible: boolean; onPress: () => void }) => {
    return (
        <ActionButton
            onPress={onPress}
            excludeFromTabOrder={!isVisible}
            UNSAFE_className={`${styles.jumpToLatest} ${isVisible ? '' : styles.jumpToLatestHidden}`}
        >
            <Text>Jump to latest</Text>
            <Icon>
                <ChevronDownLight />
            </Icon>
        </ActionButton>
    );
};
