import { SchemaRemoteTrainer } from '../../api/openapi-spec';

/** Label trainers by who maintains them: Studio or the user. */
export const connectionModeLabel = (connectionMode: SchemaRemoteTrainer['connection_mode']): string =>
    connectionMode === 'ssh' ? 'Managed by Studio' : 'Self-managed';

export const connectionModeDescription = (connectionMode: SchemaRemoteTrainer['connection_mode']): string =>
    connectionMode === 'ssh'
        ? 'Studio starts and keeps a trainer container running on the SSH host, reached through a local ' +
          'port-forward tunnel.'
        : 'You run and maintain this trainer yourself; Studio only connects to the URL it was given.';

/** The SSH host this trainer's container runs on, for display only. */
export const sshHostDisplay = (remoteTrainer: SchemaRemoteTrainer): string | undefined => {
    if (remoteTrainer.connection_mode !== 'ssh') return undefined;
    if (remoteTrainer.ssh_host_alias) return remoteTrainer.ssh_host_alias;
    if (remoteTrainer.ssh_connection) {
        return `${remoteTrainer.ssh_connection.hostname}:${remoteTrainer.ssh_connection.port}`;
    }
    return undefined;
};
