import { describe, expect, it } from 'vitest';

import { parseAwsBatchConfiguration } from './aws-batch-configuration';

const VALID = {
    schema_version: 1,
    region: 'eu-west-1',
    studio_role_arn: 'arn:aws:iam::123456789012:role/studio',
    bucket: 'jobs-bucket',
    targets: { 'g4dn.xlarge': { queue: 'arn:q', job_definition: 'arn:jd' } },
};

describe('parseAwsBatchConfiguration', () => {
    it('accepts the CloudFormation StudioConfiguration output', () => {
        const result = parseAwsBatchConfiguration(JSON.stringify(VALID));

        expect(result.error).toBeUndefined();
        expect(result.value).toEqual({ connection_mode: 'aws_batch', ...VALID });
    });

    it('rejects empty, non-JSON and non-object input', () => {
        expect(parseAwsBatchConfiguration('').error).toMatch(/Paste/);
        expect(parseAwsBatchConfiguration('{').error).toMatch(/JSON/);
        expect(parseAwsBatchConfiguration('[]').error).toMatch(/object/);
    });

    it('reports the first missing field', () => {
        const { region: _region, ...noRegion } = VALID;

        expect(parseAwsBatchConfiguration(JSON.stringify(noRegion)).error).toBe('Missing "region".');
        expect(parseAwsBatchConfiguration(JSON.stringify({ ...VALID, targets: {} })).error).toBe('Missing "targets".');
    });

    it('rejects a target without queue or job definition', () => {
        const broken = { ...VALID, targets: { 'g4dn.xlarge': { queue: 'arn:q' } } };

        expect(parseAwsBatchConfiguration(JSON.stringify(broken)).error).toMatch(/g4dn\.xlarge/);
    });

    it('rejects an unknown schema version', () => {
        expect(parseAwsBatchConfiguration(JSON.stringify({ ...VALID, schema_version: 2 })).error).toMatch(
            /schema_version/
        );
    });
});
