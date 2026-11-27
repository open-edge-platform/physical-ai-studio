// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

export interface LogTime {
    timestamp: number;
    repr: string;
}

export interface LogLevel {
    name: LogLevelName;
    no: number;
    icon: string;
}

export type LogLevelName = 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR' | 'SUCCESS' | 'CRITICAL';

export const LOG_LEVELS: LogLevelName[] = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'SUCCESS', 'CRITICAL'];

export const LOG_LEVEL_COLORS: Record<LogLevelName, string> = {
    DEBUG: 'var(--spectrum-global-color-gray-600)',
    INFO: '#ffffff',
    WARNING: 'var(--spectrum-global-color-orange-600)',
    ERROR: 'var(--spectrum-global-color-red-600)',
    SUCCESS: '#4ade80',
    CRITICAL: 'var(--spectrum-global-color-magenta-600)',
};

export interface LogProcess {
    id: number;
    name: string;
}

export interface LogThread {
    id: number;
    name: string;
}

export interface LogFile {
    name: string;
    path: string;
}

export interface LogRecord {
    elapsed: { repr: string; seconds: number };
    exception: unknown;
    extra: Record<string, unknown>;
    file: LogFile;
    function: string;
    level: LogLevel;
    line: number;
    message: string;
    module: string;
    name: string;
    process: LogProcess;
    thread: LogThread;
    time: LogTime;
}

export interface LogEntry {
    text: string;
    record: LogRecord;
}

const describe = (value: unknown): string => {
    if (typeof value === 'string') {
        return value;
    }
    try {
        return JSON.stringify(value) ?? '';
    } catch {
        return '';
    }
};

// The first line of the exception block Loguru appends, an exception group's included.
const EXCEPTION_START = /[ +]*(?:Exception Group )?Traceback \(most recent call last\):/y;

const startsException = (text: string, lineStart: number): boolean => {
    EXCEPTION_START.lastIndex = lineStart;
    return EXCEPTION_START.test(text);
};

const tracebackAfter = (text: string, lineEnd: number): string | null => {
    const traceback = text.slice(lineEnd + 1).trimEnd();
    return traceback === '' ? null : traceback;
};

/**
 * The traceback in a record's `text`. Loguru appends `\n{exception}` to the log format, so the traceback is
 * everything after the line that holds the message, whatever the format puts before or after the message.
 * The message can also appear earlier in the text (a message "ERROR" under a "{level}\n{message}" format), so
 * the occurrence followed by the start of the exception block wins, and else the first one.
 * The record's `exception` has only the type and value, not the traceback.
 */
const getTraceback = (text: unknown, message: string): string | null => {
    if (typeof text !== 'string') {
        return null;
    }
    if (message === '') {
        // An empty message is on every line, so only the start of the exception block marks where it ends.
        for (let lineEnd = text.indexOf('\n'); lineEnd !== -1; lineEnd = text.indexOf('\n', lineEnd + 1)) {
            if (startsException(text, lineEnd + 1)) {
                return tracebackAfter(text, lineEnd);
            }
        }
        return null;
    }
    let firstLineEnd = -1;
    for (let start = text.indexOf(message); start !== -1; start = text.indexOf(message, start + 1)) {
        const lineEnd = text.indexOf('\n', start + message.length);
        if (lineEnd === -1) {
            break;
        }
        if (startsException(text, lineEnd + 1)) {
            return tracebackAfter(text, lineEnd);
        }
        if (firstLineEnd === -1) {
            firstLineEnd = lineEnd;
        }
    }
    return firstLineEnd === -1 ? null : tracebackAfter(text, firstLineEnd);
};

/**
 * The exception of a record: the traceback in its `text`, or else the exception's type and value.
 * `null` when the record has none. It never throws on a malformed record.
 */
export const getExceptionText = (entry: LogEntry): string | null => {
    const { exception, message } = entry.record;
    if (exception === null || typeof exception !== 'object') {
        return null;
    }
    const traceback = getTraceback(entry.text, typeof message === 'string' ? message : '');
    if (traceback !== null) {
        return traceback;
    }
    const { type, value } = exception as { type?: unknown; value?: unknown };
    if (typeof type !== 'string') {
        return null;
    }
    const description = describe(value);
    return description === '' ? type : `${type}: ${description}`;
};

export interface LogFilters {
    levels: Set<LogLevelName>;
    searchQuery: string;
    startTime: number | null;
    endTime: number | null;
}

export const DEFAULT_LOG_FILTERS: LogFilters = {
    levels: new Set(LOG_LEVELS),
    searchQuery: '',
    startTime: null,
    endTime: null,
};

export interface LogSource {
    id: string;
    name: string;
    type: 'application' | 'worker' | 'job';
    /** ISO-8601 creation timestamp of the log file, if available. */
    created_at?: string | null;
}
