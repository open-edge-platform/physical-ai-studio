import { Badge } from '@geti-ui/ui';

import type { CatalogEntry } from './robot-families';

interface RoleBadgeProps {
    role: CatalogEntry['role'];
}

export const RoleBadge = ({ role }: RoleBadgeProps) => (
    <Badge variant={role === 'follower' ? 'info' : 'yellow'}>{role}</Badge>
);
