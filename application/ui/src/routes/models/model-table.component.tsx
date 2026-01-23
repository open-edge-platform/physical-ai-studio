import { ActionButton, Grid, Item, Key, Link, Menu, MenuTrigger, Text, View } from '@geti/ui';
import { MoreMenu } from '@geti/ui/icons';

import { SchemaModel } from '../../api/openapi-spec';
import { paths } from '../../router';
import { GRID_COLUMNS } from './constants';

import classes from './model-table.module.scss';

export const ModelHeader = () => {
    return (
        <Grid columns={GRID_COLUMNS} alignItems={'center'} width={'100%'} UNSAFE_className={classes.modelHeader}>
            <Text>Model name</Text>
            <Text>Trained</Text>
            <Text>Architecture</Text>
            <div />
            <div />
        </Grid>
    );
};

export const ModelRow = ({ model, onDelete }: { model: SchemaModel; onDelete: () => void }) => {
    const onAction = (key: Key) => {
        const action = key.toString();
        if (action === 'delete') {
            onDelete();
        }
    };

    return (
        <Grid columns={GRID_COLUMNS} alignItems={'center'} width={'100%'} UNSAFE_className={classes.modelRow}>
            <Text>{model.name}</Text>
            <Text>{new Date(model.created_at!).toLocaleString()}</Text>
            <Text>{model.policy.toUpperCase()}</Text>
            <Link
                href={paths.project.models.inference({
                    project_id: model.project_id,
                    model_id: model.id!,
                })}
            >
                Run model
            </Link>
            <View>
                <MenuTrigger>
                    <ActionButton isQuiet UNSAFE_style={{ fill: 'var(--spectrum-gray-900)' }} aria-label='options'>
                        <MoreMenu />
                    </ActionButton>
                    <Menu onAction={onAction}>
                        <Item key='delete'>Delete</Item>
                    </Menu>
                </MenuTrigger>
            </View>
        </Grid>
    );
};
