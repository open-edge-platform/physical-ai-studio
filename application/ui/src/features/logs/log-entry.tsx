// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Text, Tooltip, TooltipTrigger, View } from '@geti-ui/ui';

import { LOG_LEVEL_COLORS, type LogEntry as LogEntryType, type LogLevelName } from './log-types';

import styles from './log-viewer.module.scss';

const LogLevelText = ({ level }: { level: LogLevelName }) => {
    const color = LOG_LEVEL_COLORS[level] ?? LOG_LEVEL_COLORS.INFO;

    return (
        <Text UNSAFE_className={styles.levelTextOnly} UNSAFE_style={{ color }}>
            {level}
        </Text>
    );
};

const formatRelativeTime = (timestamp: number): string => {
    const now = Date.now() / 1000;
    const diff = now - timestamp;

    if (diff < 60) return `${Math.floor(diff)}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return `${Math.floor(diff / 86400)}d ago`;
};

const formatAbsoluteTime = (timestamp: number): string => {
    const date = new Date(timestamp * 1000);
    return date.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false,
    });
};

const LogTimestamp = ({ timestamp, repr }: { timestamp: number; repr: string }) => {
    const absoluteTime = formatAbsoluteTime(timestamp);
    const relativeTime = formatRelativeTime(timestamp);

    return (
        <TooltipTrigger delay={300}>
            <View UNSAFE_className={styles.timestamp}>
                <Text UNSAFE_className={styles.timestampText}>{absoluteTime}</Text>
            </View>
            <Tooltip>
                <Text>{repr}</Text>
                <Text UNSAFE_style={{ display: 'block', opacity: 0.7 }}>{relativeTime}</Text>
            </Tooltip>
        </TooltipTrigger>
    );
};

const LogSource = ({ module, func, line }: { module: string; func: string; line: number }) => {
    return (
        <TooltipTrigger delay={300}>
            <View UNSAFE_className={styles.source}>
                <Text UNSAFE_className={styles.sourceText}>
                    {module}:{func}
                </Text>
            </View>
            <Tooltip>
                <Text>
                    {module}:{func}:{line}
                </Text>
            </Tooltip>
        </TooltipTrigger>
    );
};

const getMessageColor = (level: LogLevelName): string => {
    return LOG_LEVEL_COLORS[level] ?? LOG_LEVEL_COLORS.INFO;
};

const TABLE_CHARS_REGEX = /[┏┓┗┛┃━┣┫┳┻╋│─├┤┬┴┼]/;

const LogMessage = ({ message, level }: { message: string; level: LogLevelName }) => {
    const color = getMessageColor(level);
    const isMultiLine = message.includes('\n');
    const hasTableChars = TABLE_CHARS_REGEX.test(message);
    const isFormattedContent = isMultiLine || hasTableChars;

    if (isFormattedContent) {
        return (
            <View UNSAFE_className={styles.messagePreformatted}>
                <pre className={styles.messagePre} style={{ color }}>
                    {message}
                </pre>
            </View>
        );
    }

    return (
        <View UNSAFE_className={styles.message}>
            <Text UNSAFE_className={styles.messageText} UNSAFE_style={{ color }}>
                {message}
            </Text>
        </View>
    );
};

export const LogEntry = ({ entry }: { entry: LogEntryType }) => {
    const { record } = entry;

    return (
        <Flex UNSAFE_className={styles.logEntry} gap='size-75' alignItems='start'>
            <LogTimestamp timestamp={record.time.timestamp} repr={record.time.repr} />
            <LogLevelText level={record.level.name} />
            <LogSource module={record.module} func={record.function} line={record.line} />
            <LogMessage message={record.message} level={record.level.name} />
        </Flex>
    );
};
