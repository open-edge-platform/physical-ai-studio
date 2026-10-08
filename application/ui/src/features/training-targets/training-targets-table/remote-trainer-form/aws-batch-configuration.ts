import { SchemaAwsBatchConnection } from '../../../../api/openapi-spec';

export type AwsBatchConfigurationParse =
    { value: SchemaAwsBatchConnection; error?: undefined } | { value?: undefined; error: string };

const isRecord = (value: unknown): value is Record<string, unknown> =>
    typeof value === 'object' && value !== null && !Array.isArray(value);

const isNonEmptyString = (value: unknown): value is string => typeof value === 'string' && value.trim() !== '';

/** Parse the `StudioConfiguration` CloudFormation output into a connection payload. */
export const parseAwsBatchConfiguration = (text: string): AwsBatchConfigurationParse => {
    if (text.trim() === '') {
        return { error: 'Paste the StudioConfiguration stack output.' };
    }

    let parsed: unknown;
    try {
        parsed = JSON.parse(text);
    } catch {
        return { error: 'Not valid JSON.' };
    }
    if (!isRecord(parsed)) {
        return { error: 'Expected a JSON object.' };
    }

    const { region, studio_role_arn, bucket, targets, schema_version } = parsed;
    if (schema_version !== undefined && schema_version !== 1) {
        return { error: `Unsupported schema_version ${String(schema_version)}.` };
    }
    for (const [key, value] of Object.entries({ region, studio_role_arn, bucket })) {
        if (!isNonEmptyString(value)) {
            return { error: `Missing "${key}".` };
        }
    }
    if (!isRecord(targets) || Object.keys(targets).length === 0) {
        return { error: 'Missing "targets".' };
    }
    const parsedTargets: SchemaAwsBatchConnection['targets'] = {};
    for (const [instanceType, target] of Object.entries(targets)) {
        if (!isRecord(target) || !isNonEmptyString(target.queue) || !isNonEmptyString(target.job_definition)) {
            return { error: `Target "${instanceType}" needs "queue" and "job_definition".` };
        }
        parsedTargets[instanceType] = { queue: target.queue, job_definition: target.job_definition };
    }

    return {
        value: {
            connection_mode: 'aws_batch',
            schema_version: 1,
            region: region as string,
            studio_role_arn: studio_role_arn as string,
            bucket: bucket as string,
            targets: parsedTargets,
        },
    };
};
