import { Flex, Heading, Text, View } from '@geti-ui/ui';

import { ReactComponent as NoProjects } from '../../../assets/illustrations/no-projects.svg';

export const NoMatchingProjects = () => {
    return (
        <View
            borderColor={'gray-600'}
            borderWidth={'thick'}
            borderRadius={'regular'}
            padding={'size-200'}
            UNSAFE_style={{ borderStyle: 'dotted' }}
        >
            <Flex gap={'size-100'} direction={'column'} alignItems={'center'} justifyContent={'center'}>
                <NoProjects />

                <Heading level={2}>No projects match your filter</Heading>

                <Text>Try adjusting your search criteria.</Text>
            </Flex>
        </View>
    );
};
