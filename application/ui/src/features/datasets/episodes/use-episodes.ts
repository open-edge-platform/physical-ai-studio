import { useQueryClient } from "@tanstack/react-query";
import { $api } from "../../../api/client";
import { useDatasetId } from "../use-dataset"

export const useDeleteEpisodeQuery = () => {
  const { dataset_id } = useDatasetId();
  const queryClient = useQueryClient();

  return $api.useMutation('delete', '/api/dataset/{dataset_id}/episodes', {
    onSuccess: (data) => {
      const query_key = [
        'get',
        '/api/dataset/{dataset_id}/episodes',
        {
          params: {
            path: {
              dataset_id,
            },
          },
        },
      ];
      queryClient.setQueryData(query_key, data);
    },
  });
}
