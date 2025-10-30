import { View } from "@geti/ui"
import { useParams } from "react-router"

export const Index = () => {
  const {project_id, model_id} = useParams();
  console.log(project_id, model_id)
  return (
    <View>
      Hello
    </View>
  )
}
