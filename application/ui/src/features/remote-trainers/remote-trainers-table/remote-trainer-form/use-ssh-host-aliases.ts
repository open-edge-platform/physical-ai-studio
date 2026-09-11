import { $api } from '../../../../api/client';
import { SchemaSshHostAliasOption } from '../../../../api/openapi-spec';

export const useSshHostAliases = () => {
    const query = $api.useQuery('get', '/api/remote-servers/aliases', {}, { retry: false });

    const aliases: SchemaSshHostAliasOption[] = query.data ?? [];

    return { aliases, isLoading: query.isLoading };
};
