import { Text, View } from '@geti-ui/ui';

import { InfoItem } from '../types';

type InfoFieldProps = {
    info: InfoItem;
};

export const InfoField = ({ info }: InfoFieldProps) => {
    const isWarning = info.variant === 'warning';
    return (
        <View
            borderWidth='thin'
            borderColor='gray-200'
            backgroundColor='gray-50'
            borderRadius='medium'
            padding='size-150'
        >
            {info.title !== undefined && info.title !== '' && (
                <Text marginBottom='size-75' UNSAFE_style={{ fontWeight: 600 }}>
                    {info.title}
                </Text>
            )}
            {isWarning && <Text marginBottom='size-50'>Warning</Text>}
            <Text>{info.text}</Text>
        </View>
    );
};
