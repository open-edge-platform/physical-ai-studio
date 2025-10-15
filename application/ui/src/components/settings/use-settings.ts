import { $api } from "../../api/client"
import { SchemaUserSettings } from "../../api/openapi-spec"

export const useSettings = (): SchemaUserSettings => {
    const {data: userSettings } = $api.useSuspenseQuery('get','/api/settings')
    return userSettings;
}