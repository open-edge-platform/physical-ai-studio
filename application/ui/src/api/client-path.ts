import { createFinalURL, createQuerySerializer } from 'openapi-fetch';
import type { HttpMethod, RequiredKeysOf } from 'openapi-typescript-helpers';

import type { paths } from './openapi-spec';

const PATH_PARAM_RE = /\{[^{}]+\}/g;

/** Extract `parameters` from any HTTP method defined on a path (they all share the same path params) */
type MethodParameters<PathItem> = {
    [M in keyof PathItem & HttpMethod]: PathItem[M] extends { parameters: infer P } ? P : never;
}[keyof PathItem & HttpMethod];

/** Build the options type: `params` is required when the parameters have required keys, optional otherwise */
type PathOptions<Params> = RequiredKeysOf<Params> extends never ? { params?: Params } : { params: Params };

export function createPathHelper<Paths extends paths>(baseUrl: string) {
    return function PATH<Path extends keyof Paths>(
        path: Path,
        ...args: RequiredKeysOf<PathOptions<MethodParameters<Paths[Path]>>> extends never
            ? [options?: PathOptions<MethodParameters<Paths[Path]>>]
            : [options: PathOptions<MethodParameters<Paths[Path]>>]
    ): string {
        if (typeof path !== 'string') {
            throw new Error('Path must be a string');
        }

        const options = args[0];
        const defaultQuerySerializer = createQuerySerializer();

        const url = createFinalURL(path, {
            baseUrl,
            params: options?.params || {},
            querySerializer: defaultQuerySerializer,
        });

        const unresolved = url.match(PATH_PARAM_RE);
        if (unresolved) {
            throw new Error(`Unresolved path parameters in "${path}": ${unresolved.join(', ')}`);
        }

        return url;
    };
}
