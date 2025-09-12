import { ProgressCircle, Flex, Heading } from '@geti/ui';

export const LoadingPage = () => {
    return (
        <Flex height={"100%"} direction={"column"} justifyContent={"center"} alignItems={"center"}>
            <ProgressCircle aria-label="Loading…" isIndeterminate />
            <Heading>Loading...</Heading>
        </Flex>
    )

}
