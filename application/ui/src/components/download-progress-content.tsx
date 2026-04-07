import { Flex, ProgressBar, ProgressCircle, Text } from '@geti-ui/ui';

type DownloadProgressContentProps = {
    isError: boolean;
    isPending: boolean;
    progress: number | null;
    errorMessage: string;
    preparingMessage: string;
};

export const DownloadProgressContent = ({
    isError,
    isPending,
    progress,
    errorMessage,
    preparingMessage,
}: DownloadProgressContentProps) => {
    if (isError) {
        return <Text>{errorMessage}</Text>;
    }

    if (!isPending) {
        return null;
    }

    return (
        <Flex direction='column' gap='size-200'>
            {progress === null ? (
                <Flex alignItems='center' gap='size-100'>
                    <ProgressCircle isIndeterminate size='S' />
                    <Text>{preparingMessage}</Text>
                </Flex>
            ) : (
                <Flex direction='column' gap='size-100'>
                    <Text>{progress}%</Text>
                    <ProgressBar value={progress} width='100%' />
                </Flex>
            )}
        </Flex>
    );
};
