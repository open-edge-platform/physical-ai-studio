import { Heading, Flex, TabList, Item, TabPanels, View, Form, TextField, NumberField, Picker, Section, Checkbox, Divider, Key, Button, ActionButton } from '@geti/ui';
import { useProjectDataContext } from "./project-config.provider"
import { SchemaRobotConfig, SchemaRobotPortInfo } from '../../../api/openapi-spec';
import { $api } from '../../../api/client';


interface RobotPropertiesProps {
    robot: SchemaRobotConfig;
    type: "leader" | "follower"
}
const RobotProperties = ({ robot, type }: RobotPropertiesProps) => {
    const { project, setProject, availableRobots, leaderCalibrations, followerCalibrations } = useProjectDataContext();

    const identifyMutation = $api.useMutation("put", "/api/hardware/identify")

    const calibrations = type === "follower" ? followerCalibrations : leaderCalibrations

    const serialIdOptions = availableRobots.map((r) => ({ id: r.serial_id, name: r.serial_id }))
    const calibrationOptions = calibrations.map((r) => ({ id: r.id, name: r.id }))

    const identify = () => {
        const portInfo = availableRobots.find((m) => m.serial_id === robot.serial_id)

        if (portInfo) {
            identifyMutation.mutate({
                body: portInfo
            })
        }
    }


    const selectRobot = (id: Key | null) => {
        const serial_id = id as string;
        const robots = project.robots.map((r) => {
            if (r.type == type) {
                return { ...r, serial_id }
            } else if (r.serial_id == serial_id) {
                return { ...r, serial_id: robot.serial_id }
            }
            return r;
        });

        setProject({ ...project, robots });
    }

    const selectCalibration = (id: Key | null) => {
        if ( id) {
            const robots = project.robots.map((r) => r.type == type ? {...robot, id: id as string} : r);
            setProject({...project, robots})
        }
    }

    return (
        <Form>
            <Heading>{type}</Heading>
            <Flex gap="size-100">
                <Picker items={serialIdOptions} selectedKey={robot.serial_id} label="Serial ID" onSelectionChange={selectRobot} flex={1}>
                    {(item) => <Item key={item.id}>{item.name}</Item>}
                </Picker>
                <Button alignSelf={"end"} isDisabled={robot.serial_id === ""} onPress={identify}>
                    Identify
                </Button>
            </Flex>
            <Picker items={calibrationOptions} label="Calibration" selectedKey={robot.id} onSelectionChange={selectCalibration}>
                {(item) => <Item>{item.name}</Item>}
            </Picker>
        </Form>
    )
}

export const RobotsView = () => {
    const { project } = useProjectDataContext();
    return (
        <>
            {project.robots.map((robot) => (
                <>
                    <RobotProperties key={robot.type} robot={robot} type={robot.type} />
                    <Divider />
                </>
            ))}


        </>
    )
}