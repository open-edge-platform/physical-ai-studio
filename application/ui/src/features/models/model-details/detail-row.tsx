import { Divider, Flex, Text, View } from '@geti-ui/ui';

const isPrimitive = (v: unknown): v is string | number | boolean | null => {
    return v === null || typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean';
};

export const DetailRow = ({ name, value }: { name: string; value: unknown }) => {
    const display = isPrimitive(value) ? String(value) : JSON.stringify(value);
    return (
        <>
            <Divider orientation='horizontal' size='S' />
            <View paddingY='size-50'>
                <Flex gap='size-200'>
                    <Text UNSAFE_style={{ flexShrink: 0, fontWeight: 'bold' }} width={'size-3000'}>
                        {name}
                    </Text>
                    <Text>{display}</Text>
                </Flex>
            </View>
        </>
    );
};
