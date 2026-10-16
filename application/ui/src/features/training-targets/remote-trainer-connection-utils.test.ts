import { SchemaRemoteTrainer } from '../../api/openapi-spec';
import { connectionModeLabel, connectionSummary, sshHostDisplay } from './remote-trainer-connection-utils';

const baseTrainer: SchemaRemoteTrainer = {
    id: 'trainer-1',
    name: 'trainer',
    connection_mode: 'direct',
    url: 'https://trainer.example.test',
    connection: { connection_mode: 'direct', url: 'https://trainer.example.test' },
};

const sshTrainer = (connection: Partial<Extract<SchemaRemoteTrainer['connection'], { connection_mode: 'ssh' }>>) =>
    ({
        ...baseTrainer,
        connection_mode: 'ssh',
        url: 'http://127.0.0.1:8001',
        connection: { connection_mode: 'ssh', ssh_remote_port: 8001, ssh_local_port: 8001, ...connection },
    }) satisfies SchemaRemoteTrainer;

const awsTrainer: SchemaRemoteTrainer = {
    ...baseTrainer,
    connection_mode: 'aws_batch',
    url: null,
    connection: {
        connection_mode: 'aws_batch',
        schema_version: 1,
        region: 'eu-west-1',
        studio_role_arn: 'arn:aws:iam::1:role/studio',
        bucket: 'jobs',
        targets: {
            'g4dn.xlarge': { queue: 'q', job_definition: 'jd' },
            'p5.4xlarge': { queue: 'q2', job_definition: 'jd2' },
        },
    },
};

describe('connectionModeLabel', () => {
    it("labels a direct trainer as 'Self-managed'", () => {
        expect(connectionModeLabel('direct')).toBe('Self-managed');
    });

    it("labels an SSH-tunneled trainer as 'Managed by Studio'", () => {
        expect(connectionModeLabel('ssh')).toBe('Managed by Studio');
    });

    it("labels an AWS Batch trainer as 'AWS Batch'", () => {
        expect(connectionModeLabel('aws_batch')).toBe('AWS Batch');
    });
});

describe('connectionSummary', () => {
    it('shows the URL for trainers that have one', () => {
        expect(connectionSummary(baseTrainer)).toBe('https://trainer.example.test');
    });

    it('shows region and instance types for AWS Batch', () => {
        expect(connectionSummary(awsTrainer)).toBe('eu-west-1 · g4dn.xlarge, p5.4xlarge');
    });
});

describe('sshHostDisplay', () => {
    it('returns undefined for a direct trainer', () => {
        expect(sshHostDisplay(baseTrainer)).toBeUndefined();
    });

    it('prefers the SSH config alias when set', () => {
        expect(sshHostDisplay(sshTrainer({ ssh_host_alias: 'gpu-box' }))).toBe('gpu-box');
    });

    it('falls back to hostname:port for a manual SSH connection', () => {
        expect(
            sshHostDisplay(
                sshTrainer({
                    ssh_connection: { hostname: '10.0.0.5', port: 2222, user: 'ec2-user', identity_file: null },
                })
            )
        ).toBe('10.0.0.5:2222');
    });
});
