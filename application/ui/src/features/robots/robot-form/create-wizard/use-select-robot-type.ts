import { useRobotCatalogQuery } from '../../robot-catalog.hooks';
import { useRobotForm } from '../provider';

export const useSelectRobotType = () => {
    const { activeType, name, setName, setActiveType } = useRobotForm();
    const { data: catalog } = useRobotCatalogQuery();

    return (type: string) => {
        if (type === activeType) {
            return;
        }

        setActiveType(type);

        const entry = catalog.find((item) => item.type === type);
        const previousSuggestedName = catalog.find((item) => item.type === activeType)?.display_name;

        if (entry !== undefined && (name.trim().length === 0 || name === previousSuggestedName)) {
            setName(entry.display_name);
        }
    };
};
