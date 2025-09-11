import { $api } from '../../api/client';
import { ErrorMessage } from '../../components/error-page/error-page';

export const Index = () => {
  const {data} = $api.useQuery('get','/api/hardware/cameras')
  console.log(data);
    return <ErrorMessage message={'Comming soon...'} />;
};
