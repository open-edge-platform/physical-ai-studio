import { describe, expect, it } from '@rstest/core';

import type { SchemaSystemInfo } from '../../api/openapi-spec';
import { createGitHubIssueUrl } from './feedback-button';

describe('createGitHubIssueUrl', () => {
    const systemInfo: SchemaSystemInfo = {
        studio_version: '0.3.0',
        git_tag: 'app/v0.3.0',
        git_sha: 'a'.repeat(40),
        git_commit_date: '2026-10-09T09:46:34Z',
        git_dirty: false,
    };

    it('adds Studio provenance without a local-changes line for a clean build', () => {
        const url = new URL(createGitHubIssueUrl(systemInfo, 'bug', 'The robot stopped.'));

        expect(url.searchParams.get('title')).toBe('[Bug]: ');
        expect(url.searchParams.get('body')).toContain('- **Commit**: ' + 'a'.repeat(40));
        expect(url.searchParams.get('body')).not.toContain('Local changes');
    });

    it('includes local changes only when they are present', () => {
        const url = new URL(createGitHubIssueUrl({ ...systemInfo, git_dirty: true }, 'feature', 'Useful feature'));

        expect(url.searchParams.get('title')).toBe('[Feature]: ');
        expect(url.searchParams.get('body')).toContain('- **Local changes**: yes');
    });
});
