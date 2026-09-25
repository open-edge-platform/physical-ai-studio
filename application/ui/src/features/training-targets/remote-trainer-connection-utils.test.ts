import { SchemaRemoteTrainer } from '../../api/openapi-spec';
import { connectionModeLabel, sshHostDisplay } from './remote-trainer-connection-utils';

const baseTrainer: SchemaRemoteTrainer = {
    id: 'trainer-1',
    name: 'trainer',
    connection_mode: 'direct',
    url: 'https://trainer.example.test',
    ssh_host_alias: null,
    ssh_connection: null,
    ssh_remote_port: null,
    ssh_local_port: null,
};

describe('connectionModeLabel', () => {
    it("labels a direct trainer as 'Self-managed'", () => {
        expect(connectionModeLabel('direct')).toBe('Self-managed');
    });

    it("labels an SSH-tunneled trainer as 'Managed by Studio'", () => {
        expect(connectionModeLabel('ssh')).toBe('Managed by Studio');
    });
});

describe('sshHostDisplay', () => {
    it('returns undefined for a direct trainer', () => {
        expect(sshHostDisplay(baseTrainer)).toBeUndefined();
    });

    it('prefers the SSH config alias when set', () => {
        expect(sshHostDisplay({ ...baseTrainer, connection_mode: 'ssh', ssh_host_alias: 'gpu-box' })).toBe('gpu-box');
    });

    it('falls back to hostname:port for a manual SSH connection', () => {
        expect(
            sshHostDisplay({
                ...baseTrainer,
                connection_mode: 'ssh',
                ssh_connection: { hostname: '10.0.0.5', port: 2222, user: 'ec2-user', identity_file: null },
            })
        ).toBe('10.0.0.5:2222');
    });
});
