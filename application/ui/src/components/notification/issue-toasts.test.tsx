import { toast } from '@geti-ui/ui';
import { rs } from '@rstest/core';
import userEvent from '@testing-library/user-event';

import { getMockedOperationalIssue } from '../../test-utils/mocks/mock-operational-issue';
import { createIssueToasts } from './issue-toasts';

describe('issue toasts', () => {
    const issue = getMockedOperationalIssue();
    let manager: ReturnType<typeof createIssueToasts>;
    let close: ReturnType<typeof rs.fn>;
    let negative: ReturnType<typeof rs.spyOn<typeof toast, 'negative'>>;

    beforeEach(() => {
        manager = createIssueToasts();
        close = rs.fn();
        negative = rs.spyOn(toast, 'negative').mockReturnValue(close);
    });
    afterEach(() => {
        rs.restoreAllMocks();
    });

    it('deduplicates across sources and clears only after the last source recovers', () => {
        for (const source of ['jobs', 'trainer:1', 'metrics:1']) manager.replaceSource(source, [issue]);
        manager.replaceSource('jobs', [{ ...issue }]);
        expect(negative).toHaveBeenCalledOnce();
        manager.replaceSource('jobs', []);
        manager.replaceSource('metrics:1', []);
        expect(close).not.toHaveBeenCalled();
        manager.replaceSource('trainer:1', []);
        expect(close).toHaveBeenCalledOnce();
        manager.replaceSource('jobs', [issue]);
        expect(negative).toHaveBeenCalledTimes(2);
    });

    it('retains issues across missing data, inconclusive checks and observer unmounts', () => {
        const first = manager.observe('trainer:1');
        const second = manager.observe('trainer:1');
        first.update([issue]);
        first.unsubscribe();
        second.update(null);
        second.update(undefined);
        second.unsubscribe();
        expect(close).not.toHaveBeenCalled();
        manager.observe('trainer:1').update([]);
        expect(close).toHaveBeenCalledOnce();
    });

    it('ignores late updates after deletion or unsubscription, preserving other sources', () => {
        const observer = manager.observe('trainer:1');
        observer.update([issue]);
        manager.replaceSource('jobs', [issue]);
        manager.removeSource('trainer:1');
        observer.update([issue]);
        expect(close).not.toHaveBeenCalled();
        manager.replaceSource('jobs', []);
        expect(close).toHaveBeenCalledOnce();
        observer.unsubscribe();
        observer.update([issue]);
        expect(negative).toHaveBeenCalledOnce();
    });

    it('updates visible content without duplicating the toast', () => {
        manager.replaceSource('jobs', [issue]);
        manager.replaceSource('jobs', [{ ...issue, detail: 'Updated instructions' }]);
        expect(close).toHaveBeenCalledOnce();
        expect(negative).toHaveBeenLastCalledWith('Updated instructions', expect.any(Object));
    });

    it('uses the latest reporting source when issue instructions change', () => {
        manager.replaceSource('jobs', [issue]);
        manager.replaceSource('trainer:1', [issue]);
        manager.replaceSource('jobs', [{ ...issue, detail: 'Updated instructions' }]);
        expect(negative).toHaveBeenLastCalledWith('Updated instructions', expect.any(Object));
    });

    it('manual dismissal survives identical polling snapshots', () => {
        manager.replaceSource('jobs', [issue]);
        negative.mock.calls[0][1]?.onClose?.();
        manager.replaceSource('jobs', [{ ...issue }]);
        expect(negative).toHaveBeenCalledOnce();
    });

    it('copies backend text and suppresses repeated snapshots until recovery', async () => {
        userEvent.setup();
        const write = rs.spyOn(navigator.clipboard, 'writeText').mockResolvedValue(undefined);
        const actionable = { ...issue, copy_text: 'provider login' };
        manager.replaceSource('jobs', [actionable]);
        const options = negative.mock.calls[0][1];
        expect(options?.actionLabel).toBe('Copy');
        await options?.onAction?.();
        expect(write).toHaveBeenCalledWith('provider login');
        expect(close).toHaveBeenCalledOnce();
        manager.replaceSource('jobs', [{ ...actionable }]);
        expect(negative).toHaveBeenCalledOnce();
        manager.replaceSource('jobs', []);
        manager.replaceSource('jobs', [actionable]);
        expect(negative).toHaveBeenCalledTimes(2);
    });

    it('copies details and reports clipboard failures', async () => {
        userEvent.setup();
        rs.spyOn(navigator.clipboard, 'writeText').mockRejectedValue(new Error('denied'));
        manager.show(issue);
        const options = negative.mock.calls[0][1];
        expect(options?.actionLabel).toBe('Copy');
        await options?.onAction?.();
        expect(negative).toHaveBeenLastCalledWith('Could not copy to clipboard.');
        expect(close).not.toHaveBeenCalled();
    });

    it('reuses an observed toast for request errors without retaining a request source', () => {
        manager.replaceSource('jobs', [issue]);
        manager.show(issue);
        expect(negative).toHaveBeenCalledOnce();
        manager.replaceSource('jobs', []);
        expect(close).toHaveBeenCalledOnce();
    });

    it('a new tab displays an issue already shown in another tab', () => {
        manager.replaceSource('jobs', [issue]);
        createIssueToasts().replaceSource('jobs', [issue]);
        expect(negative).toHaveBeenCalledTimes(2);
    });
});
