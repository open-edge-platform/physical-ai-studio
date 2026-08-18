import { describe, expect, it } from 'vitest';

import { runtimeBundleUrl } from './runtime-bundle';

describe('runtimeBundleUrl', () => {
    it('omits an empty task', () => {
        expect(
            runtimeBundleUrl({
                modelId: 'model-1',
                environmentId: 'env-1',
                backend: 'openvino',
                device: 'cpu',
                task: '  ',
            })
        ).toBe('/api/models/model-1/exports/openvino/download?environment_id=env-1&device=cpu');
    });

    it('includes a non-empty task', () => {
        expect(
            runtimeBundleUrl({
                modelId: 'model-1',
                environmentId: 'env-1',
                backend: 'openvino',
                device: 'GPU',
                task: 'pick up the cube',
            })
        ).toBe(
            '/api/models/model-1/exports/openvino/download?environment_id=env-1&device=GPU&task=pick+up+the+cube'
        );
    });
});
