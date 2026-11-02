import { SchemaRemoteTrainer } from '../../api/openapi-spec';

type ConnectionMode = SchemaRemoteTrainer['connection_mode'];

/** Label trainers by who maintains them: Studio, AWS, or the user. */
export const connectionModeLabel = (connectionMode: ConnectionMode): string => {
    switch (connectionMode) {
        case 'ssh':
            return 'Managed by Studio';
        case 'aws_batch':
            return 'AWS Batch';
        default:
            return 'Self-managed';
    }
};

export const connectionModeDescription = (connectionMode: ConnectionMode): string => {
    switch (connectionMode) {
        case 'ssh':
            return (
                'Studio starts and keeps a trainer container running on the SSH host, reached through a local ' +
                'port-forward tunnel.'
            );
        case 'aws_batch':
            return 'Each training run is submitted as an AWS Batch job; compute is provisioned on demand.';
        default:
            return 'You run and maintain this trainer yourself; Studio only connects to the URL it was given.';
    }
};

/** One-line summary of where the trainer is, for the table. */
export const connectionSummary = (remoteTrainer: SchemaRemoteTrainer): string => {
    const { connection } = remoteTrainer;
    switch (connection.connection_mode) {
        case 'aws_batch':
            return `${connection.region} · ${Object.keys(connection.targets).join(', ')}`;
        default:
            return remoteTrainer.url ?? '';
    }
};

/** The SSH host this trainer's container runs on, for display only. */
export const sshHostDisplay = (remoteTrainer: SchemaRemoteTrainer): string | undefined => {
    const { connection } = remoteTrainer;
    if (connection.connection_mode !== 'ssh') return undefined;
    if (connection.ssh_host_alias) return connection.ssh_host_alias;
    if (connection.ssh_connection) {
        return `${connection.ssh_connection.hostname}:${connection.ssh_connection.port}`;
    }
    return undefined;
};
