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

/**
 * The traceback in a record's `text`. Loguru appends `\n{exception}` to the log format, so the traceback is
 * everything after the line that holds the message, whatever the format puts before or after the message.
 * The record's `exception` has only the type and value, not the traceback.
 */
const getTraceback = (text: unknown, message: string): string | null => {
    if (typeof text !== 'string') {
        return null;
    }
    const start = text.indexOf(message);
    if (start === -1) {
        return null;
    }
    const lineEnd = text.indexOf('\n', start + message.length);
    if (lineEnd === -1) {
        return null;
    }
    const traceback = text.slice(lineEnd + 1).trimEnd();
    return traceback === '' ? null : traceback;
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
