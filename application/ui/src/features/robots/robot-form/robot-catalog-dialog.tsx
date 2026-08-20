import { useState } from 'react';

import {
    Button,
    ButtonGroup,
    Checkbox,
    Content,
    Dialog,
    DialogContainer,
    Divider,
    Flex,
    Heading,
    Link,
    SearchField,
    Text,
    View,
} from '@geti-ui/ui';
import { clsx } from 'clsx';

import { SchemaPluginResponse, SchemaPluginRobotResponse } from '../../../api/openapi-spec';
import { ReactComponent as PhysicalAIStudioLogo } from '../../../assets/icons/physicalai-studio-logo.svg';
import projectThumbnailPlaceholder from '../../../assets/project-thumbnail-placeholder.webp';
import so101BimanualThumbnail from '../../../assets/thumbnails/BimanualSO101_Follower_thumbnail.png';
import leKiwiThumbnail from '../../../assets/thumbnails/LeKiwi_Follower_thumbnail.png';
import leRobotThumbnail from '../../../assets/thumbnails/LeRobot_thumbnail.png';
import mujocoThumbnail from '../../../assets/thumbnails/MuJoCo_thumbnail.png';
import reBotArm102Thumbnail from '../../../assets/thumbnails/ReBot_Arm102_Leader_thumbnail.png';
import reBotB601Thumbnail from '../../../assets/thumbnails/ReBot_B601_DM_Follower_thumbnail.png';
import so101Thumbnail from '../../../assets/thumbnails/SO101_Leader_thumbnail.png';
import trossenBimanualThumbnail from '../../../assets/thumbnails/Trossen_Bimanual_WidowXAI_Follower_thumbnail.png';
import trossenThumbnail from '../../../assets/thumbnails/Trossen_WidowXAI_Follower_thumbnail.png';
import { InstallPluginDialog } from '../../plugins/install-plugin-dialog';
import { usePluginsQuery } from '../../plugins/plugins.hooks';
import { useRobotCatalogQuery } from '../robot-catalog.hooks';
import { useRobotForm } from './provider';

import classes from './robot-catalog-dialog.module.css';

type RobotRoleFilter = 'all' | 'follower' | 'leader';

const ROLE_CLASS_NAMES = {
    follower: classes.follower,
    leader: classes.leader,
} as const;

type CatalogManifest = {
    plugin_category: string;
    description: string;
    github_link: string;
    thumbnails?: Record<string, string>;
};

type CatalogEntry = ReturnType<typeof useRobotCatalogQuery>['data'][number];

export const CATALOG_MANIFEST: Record<string, CatalogManifest> = {
    SO101: {
        plugin_category: 'SO101',
        description: 'LeRobot SO-101 arms for learning from demonstration.',
        github_link: 'https://github.com/huggingface/lerobot',
        thumbnails: {
            SO101_Follower: so101Thumbnail,
            SO101_Leader: so101Thumbnail,
            BimanualSO101_Follower: so101BimanualThumbnail,
            BimanualSO101_Leader: so101BimanualThumbnail,
        },
    },
    Trossen: {
        plugin_category: 'Trossen',
        description: 'Trossen Robotics WidowX AI arm integrations.',
        github_link: 'https://github.com/TrossenRobotics',
        thumbnails: {
            Trossen_WidowXAI_Follower: trossenThumbnail,
            Trossen_WidowXAI_Leader: trossenThumbnail,
            Trossen_Bimanual_WidowXAI_Follower: trossenBimanualThumbnail,
            Trossen_Bimanual_WidowXAI_Leader: trossenBimanualThumbnail,
        },
    },
    ReBot: {
        plugin_category: 'ReBot',
        description: 'ReBot B601 and Arm102 robot integrations.',
        github_link: 'https://github.com/open-edge-platform/physical-ai-rebot-b601-plugin',
        thumbnails: {
            ReBot_B601_DM_Follower: reBotB601Thumbnail,
            ReBot_Arm102_Leader: reBotArm102Thumbnail,
        },
    },
    LeRobot: {
        plugin_category: 'LeRobot',
        description: 'Robot and teleoperator configurations discovered from LeRobot.',
        github_link: 'https://github.com/huggingface/lerobot',
    },
    LeKiwi: {
        plugin_category: 'LeKiwi',
        description: 'LeKiwi mobile manipulator integration.',
        github_link: 'https://github.com/huggingface/lerobot',
        thumbnails: {
            LeKiwi_Follower: leKiwiThumbnail,
            LeKiwi_Leader: leKiwiThumbnail,
        },
    },
    MuJoCo: {
        plugin_category: 'MuJoCo',
        description: 'MuJoCo-backed SO-101 simulation integration.',
        github_link: 'https://github.com/google-deepmind/mujoco',
        thumbnails: {
            MuJoCo_SO101_Follower: mujocoThumbnail,
        },
    },
};

const RobotCard = ({
    entry,
    category,
    activeType,
    onSelect,
    isAvailable = false,
}: {
    entry: CatalogEntry | SchemaPluginRobotResponse;
    category: string;
    activeType: string | undefined;
    onSelect: () => void;
    isAvailable?: boolean;
}) => {
    const thumbnail =
        category === 'LeRobot'
            ? undefined
            : (CATALOG_MANIFEST[category]?.thumbnails?.[entry.type] ??
              ('preview_thumbnail' in entry ? entry.preview_thumbnail : undefined));

    return (
        <Button
            variant={!isAvailable && activeType === entry.type ? 'accent' : 'secondary'}
            onPress={onSelect}
            UNSAFE_className={clsx(classes.card, isAvailable && classes.availableCard)}
            UNSAFE_style={{ alignItems: 'flex-start', justifyContent: 'flex-start' }}
        >
            <div className={classes.cardContent}>
                <div className={classes.thumbnailArea}>
                    {thumbnail ? (
                        <img className={classes.thumbnail} src={thumbnail} alt='' />
                    ) : (
                        <div className={classes.thumbnailFallback}>
                            <img src={projectThumbnailPlaceholder} width={124} height={124} alt='' aria-hidden />
                        </div>
                    )}
                </div>
                <span aria-label={entry.role} className={clsx(classes.role, ROLE_CLASS_NAMES[entry.role])} />
                <div className={classes.cardDetails}>
                    <Text>{entry.display_name}</Text>
                    {isAvailable && <Text UNSAFE_className={classes.notInstalled}>Not installed</Text>}
                </div>
            </div>
        </Button>
    );
};

export const RobotCatalogDialog = ({ close }: { close: () => void }) => {
    const { activeType, setActiveType } = useRobotForm();
    const catalog = useRobotCatalogQuery();
    const plugins = usePluginsQuery();
    const [role, setRole] = useState<RobotRoleFilter>('all');
    const [showExternal, setShowExternal] = useState(true);
    const [search, setSearch] = useState('');
    const [installPlugin, setInstallPlugin] = useState<SchemaPluginResponse | undefined>();
    const normalizedSearch = search.trim().toLocaleLowerCase();
    const entries = catalog.data.filter(
        (entry) =>
            (role === 'all' || entry.role === role) &&
            (showExternal || entry.source !== 'external') &&
            (normalizedSearch === '' ||
                entry.display_name.toLocaleLowerCase().includes(normalizedSearch) ||
                entry.category.toLocaleLowerCase().includes(normalizedSearch))
    );
    const categories = new Map<string, typeof entries>();
    entries.forEach((entry) => {
        categories.set(entry.category, [...(categories.get(entry.category) ?? []), entry]);
    });

    const availablePlugins = plugins.data.filter(
        (plugin) => !plugin.installed && plugin.robots.length > 0 && (showExternal || plugin.source !== 'external')
    );

    const selectRobot = (type: string) => {
        setActiveType(type);
        close();
    };

    const openPlugins = (plugin: SchemaPluginResponse) => {
        setInstallPlugin(plugin);
    };

    const matchesFilters = (displayName: string, category: string, robotRole: string) =>
        (role === 'all' || robotRole === role) &&
        (normalizedSearch === '' ||
            displayName.toLocaleLowerCase().includes(normalizedSearch) ||
            category.toLocaleLowerCase().includes(normalizedSearch));

    return (
        <Dialog size='L' width={'100%'} height='100%'>
            <Heading>
                <Flex width='100%' justifyContent={'space-between'}>
                    <span>Select robot type</span>
                    <Flex alignItems={'center'} gap='size-200'>
                        <ButtonGroup aria-label='Robot role filter'>
                            {(['all', 'follower', 'leader'] as const).map((filter) => (
                                <Button
                                    key={filter}
                                    variant={role === filter ? 'accent' : 'secondary'}
                                    onPress={() => setRole(filter)}
                                    UNSAFE_className={
                                        filter === 'all'
                                            ? undefined
                                            : clsx(classes.filterRole, ROLE_CLASS_NAMES[filter])
                                    }
                                >
                                    {filter === 'all' ? 'All roles' : `${filter[0].toUpperCase()}${filter.slice(1)}s`}
                                </Button>
                            ))}
                        </ButtonGroup>
                        <Checkbox isSelected={showExternal} onChange={setShowExternal}>
                            Show external plugins
                        </Checkbox>
                        <SearchField
                            aria-label='Search robot types'
                            placeholder='Search robots'
                            value={search}
                            onChange={setSearch}
                            onClear={() => setSearch('')}
                            width='size-3600'
                        />
                    </Flex>
                </Flex>
            </Heading>
            <Divider />
            <Content UNSAFE_className={classes.content}>
                {[...categories].map(([category, robots]) => (
                    <View key={category}>
                        <Flex alignItems='baseline' gap='size-150'>
                            <Heading level={3}>{category}</Heading>
                            <Text>{CATALOG_MANIFEST[category]?.description ?? 'Robot integration plugin.'}</Text>
                            {CATALOG_MANIFEST[category] && (
                                <Link href={CATALOG_MANIFEST[category].github_link} target='_blank'>
                                    GitHub
                                </Link>
                            )}
                        </Flex>
                        <Flex gap='size-200' UNSAFE_className={classes.robotRow}>
                            {robots.map((entry) => (
                                <RobotCard
                                    key={entry.type}
                                    entry={entry}
                                    category={category}
                                    activeType={activeType}
                                    onSelect={() => selectRobot(entry.type)}
                                />
                            ))}
                        </Flex>
                    </View>
                ))}
                {availablePlugins.length > 0 && (
                    <View>
                        <Flex alignItems='baseline' gap='size-150'>
                            <Heading level={3}>Available plugins</Heading>
                            <Text>Install a plugin to add these robots.</Text>
                        </Flex>
                        <Flex direction='column' gap='size-200'>
                            {availablePlugins.map((plugin) => {
                                const robots = plugin.robots.filter((robot) =>
                                    matchesFilters(robot.display_name, plugin.category, robot.role)
                                );
                                if (robots.length === 0) return null;
                                return (
                                    <View key={plugin.id}>
                                        <Flex alignItems='baseline' gap='size-150'>
                                            <Heading level={4}>{plugin.category}</Heading>
                                            <Text>{plugin.description}</Text>
                                            <Button variant='secondary' onPress={() => openPlugins(plugin)}>
                                                Install plugin
                                            </Button>
                                        </Flex>
                                        <Flex gap='size-200' UNSAFE_className={classes.robotRow}>
                                            {robots.map((robot) => (
                                                <RobotCard
                                                    key={robot.type}
                                                    entry={robot}
                                                    category={plugin.category}
                                                    activeType={undefined}
                                                    isAvailable
                                                    onSelect={() => openPlugins(plugin)}
                                                />
                                            ))}
                                        </Flex>
                                    </View>
                                );
                            })}
                        </Flex>
                    </View>
                )}
                {entries.length === 0 && <Text>No robots match the selected filters.</Text>}
            </Content>
            <DialogContainer onDismiss={() => setInstallPlugin(undefined)}>
                {installPlugin && (
                    <InstallPluginDialog plugin={installPlugin} close={() => setInstallPlugin(undefined)} />
                )}
            </DialogContainer>
        </Dialog>
    );
};
