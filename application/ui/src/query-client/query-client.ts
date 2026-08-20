import { matchQuery, MutationCache, Query, QueryClient } from '@tanstack/react-query';

import { MutationMeta } from './query-client.interface';

declare module '@tanstack/react-query' {
    interface Register {
        mutationMeta: MutationMeta;
    }
}

export const createQueryClient = (): QueryClient => {
    const client = new QueryClient({
        defaultOptions: {
            queries: {
                gcTime: 30 * 60 * 1000,
                staleTime: 5 * 60 * 1000,
                networkMode: 'always',
                retry: false,
            },
            mutations: {
                networkMode: 'always',
                retry: false,
            },
        },
        mutationCache: new MutationCache({
            onSuccess: (_data, _variables, _context, mutation): void | Promise<void> => {
                if (mutation.meta?.skipInvalidation) return;

                // Fire-and-forget invalidation
                client.invalidateQueries({
                    predicate: (query: Query): boolean => {
                        return (
                            mutation.meta?.invalidates?.some((queryKey) => {
                                return matchQuery({ queryKey }, query);
                            }) ?? false
                        );
                    },
                });

                // Optionally await specific query invalidations
                if (mutation.meta?.awaits && mutation.meta.awaits.length > 0) {
                    return client.invalidateQueries(
                        {
                            predicate: (query) => {
                                return (
                                    mutation.meta?.awaits?.some((queryKey) => matchQuery({ queryKey }, query), {}) ??
                                    false
                                );
                            },
                        },
                        { cancelRefetch: false }
                    );
                }
            },
        }),
    });

    return client;
};

export const queryClient = createQueryClient();
