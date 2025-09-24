import { ActionButton, Item, Key, Menu, MenuTrigger } from '@geti/ui';
import { MoreMenu } from '@geti/ui/icons';

interface MenuActionsProps {
    onAction: (key: Key) => void
}
export const MenuActions = ({onAction}: MenuActionsProps) => {
    return (
        <MenuTrigger>
            <ActionButton isQuiet UNSAFE_style={{ fill: 'var(--spectrum-gray-900)' }}>
                <MoreMenu />
            </ActionButton>
            <Menu onAction={onAction}>
                <Item key={"export"}>Export</Item>
                <Item key={"duplicate"}>Duplicate</Item>
                <Item key={"delete"}>Delete</Item>
            </Menu>
        </MenuTrigger>
    );
};
