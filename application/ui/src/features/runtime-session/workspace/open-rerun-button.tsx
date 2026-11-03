import { Button, Flex, Text } from '@geti-ui/ui';

export const OpenRerunButton = ({ url }: { url: string | null | undefined }) => (
    <Flex direction='column' gap='size-100'>
        <Button variant='secondary' href={url ?? undefined} target='_blank' rel='noopener noreferrer' isDisabled={!url}>
            Open Rerun
        </Button>
        <Text>Opens Rerun’s hosted web viewer. Local streams require a browser on the backend machine.</Text>
    </Flex>
);
