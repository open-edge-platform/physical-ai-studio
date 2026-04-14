export const getFilenameFromContentDisposition = (
    contentDisposition: string | null,
    fallbackFilename: string
): string => {
    if (!contentDisposition) return fallbackFilename;

    const utfFilenameMatch = contentDisposition.match(/filename\*=UTF-8''([^;]+)/i);
    if (utfFilenameMatch?.[1]) {
        return decodeURIComponent(utfFilenameMatch[1]);
    }

    const filenameMatch = contentDisposition.match(/filename="?([^";]+)"?/i);
    return filenameMatch?.[1] ?? fallbackFilename;
};

export const isAbortError = (error: unknown): boolean => {
    return error instanceof DOMException && error.name === 'AbortError';
};

/**
 * Reads a streaming HTTP response and reconstructs the final archive Blob.
 *
 * Instead of calling `response.blob()`, we manually read the stream using a
 * `ReadableStreamDefaultReader`. This allows us to track download progress
 * when the `content-length` header is available.
 */
export const getArchiveBlobFromResponse = async (
    response: Response,
    onProgress: (progress: number | null) => void
): Promise<Blob> => {
    const contentLength = response.headers.get('content-length');
    const totalBytes = contentLength ? Number(contentLength) : null;

    if (!response.body) {
        return response.blob();
    }

    const reader = response.body.getReader();
    const chunks: Uint8Array<ArrayBuffer>[] = [];

    let downloadedBytes = 0;

    while (true) {
        const { done, value } = await reader.read();

        if (done) break;

        if (value) {
            chunks.push(value);
            downloadedBytes += value.length;

            if (totalBytes && Number.isFinite(totalBytes)) {
                const percent = Math.round((downloadedBytes / totalBytes) * 100);
                onProgress(percent);
            }
        }
    }

    return new Blob(chunks, { type: 'application/zip' });
};
