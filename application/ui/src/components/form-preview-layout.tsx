import { ComponentProps, ReactNode, Suspense } from 'react';

import { Flex, Grid, Loading, minmax, View } from '@geti-ui/ui';

type FormPreviewLayoutProps = {
    form: ReactNode;
    preview: ReactNode;
    previewProps?: Omit<ComponentProps<typeof View>, 'children' | 'gridArea' | 'minHeight' | 'minWidth' | 'overflow'>;
};

const CenteredLoading = () => {
    return (
        <Flex width='100%' height='100%' alignItems='center' justifyContent='center'>
            <Loading mode='inline' />
        </Flex>
    );
};

export const FormPreviewLayout = ({ form, preview, previewProps }: FormPreviewLayoutProps) => {
    return (
        <Grid
            areas={['form preview']}
            columns={[minmax('size-6000', 'auto'), minmax(0, '1fr')]}
            rows={[minmax(0, '1fr')]}
            height='100%'
            minHeight={0}
        >
            <View gridArea='form' backgroundColor='gray-100' padding='size-400' minHeight={0} overflow='auto'>
                <Suspense fallback={<CenteredLoading />}>{form}</Suspense>
            </View>
            <View gridArea='preview' {...previewProps} minHeight={0} minWidth={0} overflow='hidden'>
                {preview}
            </View>
        </Grid>
    );
};
