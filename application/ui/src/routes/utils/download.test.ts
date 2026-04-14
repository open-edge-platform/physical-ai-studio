import { describe, expect, it, vi } from 'vitest';

import { getArchiveBlobFromResponse, getFilenameFromContentDisposition, isAbortError } from './download';

describe('getFilenameFromContentDisposition', () => {
    it('returns fallback for null header', () => {
        expect(getFilenameFromContentDisposition(null, 'fallback.zip')).toBe('fallback.zip');
    });

    it('extracts quoted filename', () => {
        expect(getFilenameFromContentDisposition('attachment; filename="dataset.zip"', 'fallback.zip')).toBe(
            'dataset.zip'
        );
    });

    it('extracts unquoted filename', () => {
        expect(getFilenameFromContentDisposition('attachment; filename=dataset.zip', 'fallback.zip')).toBe(
            'dataset.zip'
        );
    });

    it('prefers and decodes filename* when present', () => {
        expect(
            getFilenameFromContentDisposition(
                "attachment; filename=plain.zip; filename*=UTF-8''My%20Dataset%40v1.zip",
                'fallback.zip'
            )
        ).toBe('My Dataset@v1.zip');
    });
});

describe('isAbortError', () => {
    it('returns true for AbortError DOMException', () => {
        expect(isAbortError(new DOMException('aborted', 'AbortError'))).toBe(true);
    });

    it('returns false for non-abort errors', () => {
        expect(isAbortError(new DOMException('network', 'NetworkError'))).toBe(false);
        expect(isAbortError(new Error('boom'))).toBe(false);
        expect(isAbortError('AbortError')).toBe(false);
    });
});

const makeStreamingResponse = (chunks: Uint8Array[], contentLength?: number): Response => {
    const stream = new ReadableStream<Uint8Array>({
        start(controller) {
            for (const chunk of chunks) controller.enqueue(chunk);
            controller.close();
        },
    });

    const headers = new Headers({ 'content-type': 'application/zip' });
    if (contentLength !== undefined) {
        headers.set('content-length', String(contentLength));
    }

    return new Response(stream, { status: 200, headers });
};

describe('getArchiveBlobFromResponse', () => {
    it('returns a blob with joined content', async () => {
        const response = makeStreamingResponse([new Uint8Array([1, 2]), new Uint8Array([3, 4])]);
        const blob = await getArchiveBlobFromResponse(response, vi.fn());

        expect(blob.type).toBe('application/zip');
        expect(blob.size).toBe(4);
    });

    it('reports progress when content-length is available', async () => {
        const onProgress = vi.fn();
        const response = makeStreamingResponse([new Uint8Array(50), new Uint8Array(50)], 100);

        await getArchiveBlobFromResponse(response, onProgress);

        expect(onProgress).toHaveBeenNthCalledWith(1, 50);
        expect(onProgress).toHaveBeenNthCalledWith(2, 100);
    });

    it('does not report progress without content-length', async () => {
        const onProgress = vi.fn();
        const response = makeStreamingResponse([new Uint8Array([1])]);

        await getArchiveBlobFromResponse(response, onProgress);

        expect(onProgress).not.toHaveBeenCalled();
    });

    it('falls back to response.blob when body is null', async () => {
        const expectedBlob = new Blob([new Uint8Array([9])], { type: 'application/zip' });
        const response = {
            body: null,
            headers: new Headers(),
            blob: vi.fn().mockResolvedValue(expectedBlob),
        } as unknown as Response;

        const blob = await getArchiveBlobFromResponse(response, vi.fn());
        expect(blob).toBe(expectedBlob);
    });
});
