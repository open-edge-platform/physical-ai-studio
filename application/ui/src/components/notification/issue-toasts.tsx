import { useEffect } from 'react';

import { toast } from '@geti-ui/ui';

import { getApiErrorMessage } from '../../api/errors';
import { SchemaOperationalIssue } from '../../api/openapi-spec';

interface Source {
    issues: Map<string, SchemaOperationalIssue>;
    observers: number;
    revision: number;
}

interface Notification {
    issue: SchemaOperationalIssue;
    close: () => void;
    dismissed: boolean;
}

export const isOperationalIssue = (error: unknown): error is SchemaOperationalIssue =>
    error !== null &&
    typeof error === 'object' &&
    'id' in error &&
    typeof error.id === 'string' &&
    'title' in error &&
    typeof error.title === 'string' &&
    'detail' in error &&
    typeof error.detail === 'string';

const renderIssue = (issue: Pick<SchemaOperationalIssue, 'detail' | 'copy_text'>, onDismiss = () => {}) => {
    const close = toast.negative(issue.detail, {
        actionLabel: 'Copy',
        shouldCloseOnAction: false,
        onClose: onDismiss,
        onAction: async () => {
            try {
                await navigator.clipboard.writeText(issue.copy_text ?? issue.detail);
                onDismiss();
                close();
            } catch {
                toast.negative('Could not copy to clipboard.');
            }
        },
    });
    return close;
};

export const createIssueToasts = () => {
    const sources = new Map<string, Source>();
    const notifications = new Map<string, Notification>();
    let revision = 0;

    const display = (issue: SchemaOperationalIssue) => {
        const notification: Notification = { issue, close: () => {}, dismissed: false };
        notification.close = renderIssue(issue, () => {
            notification.dismissed = true;
        });
        notifications.set(issue.id, notification);
    };
    const reconcile = () => {
        const active = new Map<string, SchemaOperationalIssue>();
        for (const source of [...sources.values()].sort((a, b) => a.revision - b.revision)) {
            for (const [id, issue] of source.issues) active.set(id, issue);
        }
        for (const [id, notification] of notifications) {
            if (!active.has(id)) {
                notification.close();
                notifications.delete(id);
            }
        }
        for (const [id, issue] of active) {
            const notification = notifications.get(id);
            if (!notification) display(issue);
            else if (JSON.stringify(notification.issue) !== JSON.stringify(issue)) {
                notification.issue = issue;
                if (!notification.dismissed) {
                    notification.close();
                    display(issue);
                }
            }
        }
    };
    const getSource = (id: string) => {
        let source = sources.get(id);
        if (!source) {
            source = { issues: new Map(), observers: 0, revision: 0 };
            sources.set(id, source);
        }
        return source;
    };
    const update = (source: Source, issues: SchemaOperationalIssue[] | null | undefined) => {
        if (issues == null) return;
        source.issues = new Map(issues.map((issue) => [issue.id, issue]));
        source.revision = ++revision;
        reconcile();
    };
    return {
        replaceSource: (id: string, issues: SchemaOperationalIssue[] | null | undefined) => {
            if (issues != null) update(getSource(id), issues);
        },
        removeSource: (id: string) => {
            sources.delete(id);
            reconcile();
        },
        hasIssues: (id: string) => (sources.get(id)?.issues.size ?? 0) > 0,
        observe: (id: string) => {
            const source = getSource(id);
            source.observers++;
            let subscribed = true;
            return {
                update: (issues: SchemaOperationalIssue[] | null | undefined) => {
                    if (subscribed && sources.get(id) === source) update(source, issues);
                },
                unsubscribe: () => {
                    if (!subscribed) return;
                    subscribed = false;
                    source.observers--;
                    if (sources.get(id) === source && !source.observers && !source.issues.size) sources.delete(id);
                },
            };
        },
        show: (issue: SchemaOperationalIssue) => {
            const existing = notifications.get(issue.id);
            if (existing) {
                if (existing.dismissed) display(issue);
                return () => {};
            }
            return renderIssue(issue);
        },
    };
};

export const issueToasts = createIssueToasts();

export const showIssueError = (error: unknown) => {
    if (isOperationalIssue(error)) return issueToasts.show(error);
    return renderIssue({ detail: getApiErrorMessage(error) ?? 'The operation failed.' });
};

export const IssueErrorAlert = ({ error }: { error: unknown }) => {
    useEffect(() => showIssueError(error), [error]);
    return null;
};
