import { Suspense, useState } from 'react';

import {
    ActionButton,
    Button,
    ButtonGroup,
    Content,
    Dialog,
    DialogTrigger,
    Divider,
    Flex,
    Heading,
    Icon,
    Item,
    Loading,
    Picker,
    Text,
    TextArea,
    Tooltip,
    TooltipTrigger,
} from '@geti-ui/ui';
import { ExternalLinkIcon, HelpIcon } from '@geti-ui/ui/icons';

import { $api } from '../../api/client';
import type { SchemaSystemInfo } from '../../api/openapi-spec';

const GITHUB_ISSUES_URL = 'https://github.com/open-edge-platform/physical-ai-studio/issues/new';

type IssueType = 'bug' | 'feature';

const issueTypeOptions = [
    { id: 'bug', label: 'Bug report' },
    { id: 'feature', label: 'Feature request' },
] as const;

const sanitizeDescription = (description: string): string => {
    const cleaned = description.replace(/[^\x09\x0A\x0D\x20-\x7E\xA0-\uFFFF]/g, '').trim();
    const maxLength = 4000;

    return cleaned.length <= maxLength ? cleaned : `${cleaned.slice(0, maxLength)}\n\n[Description truncated]`;
};

const buildStudioInformation = (systemInfo: SchemaSystemInfo): string[] => {
    const lines = [`- **Version**: ${systemInfo.studio_version}`];
    if (systemInfo.git_tag) lines.push(`- **Source tag**: ${systemInfo.git_tag}`);
    if (systemInfo.git_sha) lines.push(`- **Commit**: ${systemInfo.git_sha}`);
    if (systemInfo.git_commit_date) lines.push(`- **Commit date**: ${systemInfo.git_commit_date}`);
    if (systemInfo.git_dirty) lines.push('- **Local changes**: yes');

    return lines;
};

export const createGitHubIssueUrl = (
    systemInfo: SchemaSystemInfo,
    issueType: IssueType,
    description: string
): string => {
    const isBug = issueType === 'bug';
    const body = `## Studio information
${buildStudioInformation(systemInfo).join('\n')}

## ${isBug ? 'Bug description' : 'Feature description'}
${sanitizeDescription(description) || '_Please describe your feedback_'}
`;
    const params = new URLSearchParams({
        title: isBug ? '[Bug]: ' : '[Feature]: ',
        body,
    });

    return `${GITHUB_ISSUES_URL}?${params.toString()}`;
};

const openGitHubIssue = ({
    issueType,
    description,
    systemInfo,
}: {
    issueType: IssueType;
    description: string;
    systemInfo: SchemaSystemInfo;
}): void => {
    window.open(createGitHubIssueUrl(systemInfo, issueType, description), '_blank', 'noopener,noreferrer');
};

const FeedbackDialog = ({ close }: { close: () => void }) => {
    const [issueType, setIssueType] = useState<IssueType>('bug');
    const [description, setDescription] = useState('');
    const { data: systemInfo, isPending: isSystemInfoPending } = $api.useSuspenseQuery('get', '/api/system/info');

    const studioVersionText = systemInfo
        ? [
              `Physical AI Studio v${systemInfo.studio_version}`,
              systemInfo.git_tag,
              systemInfo.git_sha?.slice(0, 8),
              systemInfo.git_commit_date,
          ]
              .filter(Boolean)
              .join(' · ')
        : 'Studio version information will be retrieved when the issue is opened.';

    return (
        <Dialog>
            <Heading>Submit feedback</Heading>
            <Divider />
            <Content>
                <Flex direction='column' gap='size-200'>
                    <Text>
                        Report a bug or suggest a feature. Studio version and source revision details will be included
                        in the GitHub issue draft.
                    </Text>
                    <Picker
                        label='Feedback type'
                        selectedKey={issueType}
                        onSelectionChange={(key) => setIssueType(key as IssueType)}
                        width='100%'
                    >
                        {issueTypeOptions.map((option) => (
                            <Item key={option.id}>{option.label}</Item>
                        ))}
                    </Picker>
                    <TextArea
                        label='Description'
                        placeholder={
                            issueType === 'bug'
                                ? 'Describe what happened and what you expected to happen...'
                                : 'Describe the feature you would like to see...'
                        }
                        value={description}
                        onChange={setDescription}
                        width='100%'
                        height='size-1600'
                    />
                    <Text UNSAFE_style={{ color: 'var(--spectrum-global-color-gray-600)' }}>
                        {isSystemInfoPending ? 'Loading Studio version information…' : studioVersionText}
                    </Text>
                </Flex>
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={close}>
                    Cancel
                </Button>
                <Button
                    variant='accent'
                    onPress={() => {
                        openGitHubIssue({ issueType, description, systemInfo });
                        close();
                    }}
                >
                    <Icon>
                        <ExternalLinkIcon />
                    </Icon>
                    <Text>Open GitHub issue</Text>
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};

export const FeedbackButton = () => (
    <DialogTrigger type='modal'>
        <TooltipTrigger delay={300}>
            <ActionButton isQuiet aria-label='Submit feedback'>
                <Icon>
                    <HelpIcon />
                </Icon>
                <Text>Feedback</Text>
            </ActionButton>
            <Tooltip>Submit feedback</Tooltip>
        </TooltipTrigger>
        {(close) => (
            <Suspense fallback={<Loading />}>
                <FeedbackDialog close={close} />
            </Suspense>
        )}
    </DialogTrigger>
);
