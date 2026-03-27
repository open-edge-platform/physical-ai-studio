import { render, screen } from '@testing-library/react';

import { DownloadProgressContent } from './download-progress-content';

describe('DownloadProgressContent', () => {
    const baseProps = {
        isError: false,
        isPending: false,
        progress: null,
        errorMessage: 'Download failed',
        preparingMessage: 'Preparing download...',
    };

    it('renders nothing when idle', () => {
        const { container } = render(<DownloadProgressContent {...baseProps} />);
        expect(container).toBeEmptyDOMElement();
    });

    it('renders error message when error', () => {
        render(<DownloadProgressContent {...baseProps} isError />);
        expect(screen.getByText('Download failed')).toBeInTheDocument();
    });

    it('renders preparing state when pending and progress is null', () => {
        render(<DownloadProgressContent {...baseProps} isPending progress={null} />);
        expect(screen.getByText('Preparing download...')).toBeInTheDocument();
    });

    it('renders progress value when pending and progress is numeric', () => {
        render(<DownloadProgressContent {...baseProps} isPending progress={42} />);
        expect(screen.getByText('42%')).toBeInTheDocument();
    });

    it('prioritizes error over pending state', () => {
        render(<DownloadProgressContent {...baseProps} isError isPending progress={42} />);
        expect(screen.getByText('Download failed')).toBeInTheDocument();
        expect(screen.queryByText('42%')).not.toBeInTheDocument();
    });
});
