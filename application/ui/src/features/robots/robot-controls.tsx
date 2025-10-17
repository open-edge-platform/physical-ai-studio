import { useEffect, useRef, useState } from 'react';

import { Button, Flex, Heading, Slider, Text, View } from '@geti/ui';
import { useHover } from 'react-aria';
import * as THREE from 'three';
import { URDFBase, URDFJoint, URDFLink, URDFRobot, URDFVisual } from 'urdf-loader';

import { useAction } from './action-context';

type JointControlProps = {
    joint: URDFJoint;
    onChange: (name: string, value: number) => void;
};

const JointControl = ({ joint, onChange }: JointControlProps) => {
    const [value, setValue] = useState(joint.jointValue.at(0) || 0);

    const handleChange = (newValue: number) => {
        setValue(newValue);
        onChange(joint.name, newValue);
    };

    const min: number = joint.limit.lower || -Math.PI;
    const max: number = joint.limit.upper || Math.PI;

    return (
        <View flexGrow={1}>
            <Slider
                width='100%'
                isFilled
                name={joint.name.substring(joint.name.indexOf('_') + 1)}
                label={`${joint.name} - ${joint.name.substring(joint.name.indexOf('_') + 1)}`}
                getValueLabel={(value) => {
                    const degrees = (90 * value) / (Math.PI * 2);
                    return `${degrees.toFixed(2)}°`;
                }}
                minValue={min}
                maxValue={max}
                step={0.01}
                value={value}
                onChange={handleChange}
            />
        </View>
    );
};

export const JointControls = ({ model }: { model: URDFRobot }) => {
    const joints = Object.values(model.joints).filter((joint) => joint.jointType !== 'fixed');

    console.log('movable joints', joints);

    const handleJointChange = (name: string, value: number) => {
        model.joints[name]?.setJointValue(value);
    };

    const parts = Object.groupBy(joints, (joint) => joint.name?.split('_')?.at(0) ?? 'Other');

    return (
        <div className='robot-controls'>
            <h3>Joint Controls</h3>

            {joints.length === 0 && <p>No controllable joints available</p>}

            <Flex direction='column' gap='size-200' width='100%'>
                {Object.keys(parts)
                    .reverse()
                    .map((part) => {
                        return (
                            <View width='100%' key={part}>
                                <Heading level={4}>{part}</Heading>

                                <Flex direction={'row'} width='100%' justifyContent='space-between' gap='size-200'>
                                    {parts[part]?.map((joint) => (
                                        <JointControl key={joint.name} joint={joint} onChange={handleJointChange} />
                                    ))}
                                </Flex>
                            </View>
                        );
                    })}
            </Flex>

            <Flex direction='column' gap='size-200' isHidden>
                {joints.length === 0 ? (
                    <p>No controllable joints available</p>
                ) : (
                    joints.map((joint) => <JointControl key={joint.name} joint={joint} onChange={handleJointChange} />)
                )}
            </Flex>
        </div>
    );
};

type LinkControlProps = {
    link: URDFLink;
    name: string;
};
const LinkControl = ({ link, name }: LinkControlProps) => {
    const original = useRef<THREE.MeshPhongMaterial | null>(null);

    const { hoverProps } = useHover({
        onHoverChange: (isHovering) => {
            //console.log(isHovering, link, link.children);

            //link.traverse((x) => console.log(x.urdfName, x.type, x.name));
            const traverse = (c: URDFBase) => {
                //console.log(c.urdfName, c.type, c.name, c.isURDFLink);

                if (c.type === 'Mesh' && c?.parent?.parent?.isURDFLink) {
                    if (isHovering) {
                        console.log(c, c.type, c.parent.type);
                    }

                    const mesh = c as THREE.Mesh;
                    if (isHovering === false && original.current !== null) {
                        mesh.material = original.current;
                    } else {
                        original.current = mesh.material;
                        mesh.material = new THREE.MeshPhongMaterial({
                            shininess: 10,
                            color: '#ff00ff', //this.highlightColor,
                            emissive: '#ff00ff', //this.highlightColor,
                            emissiveIntensity: 0.25,
                        });
                    }
                }

                for (let i = 0; i < c.children.length; i++) {
                    traverse(c.children[i]);
                }
            };
            traverse(link);
        },
    });

    return (
        <div {...hoverProps}>
            <View padding='size-100' backgroundColor={'blue-400'}>
                {name} - {link.name}
            </View>
        </div>
    );
};

const ModelTree = ({ model }: { model: THREE.Object3D }) => {
    const original = useRef<THREE.MeshPhongMaterial | null>(null);

    const { hoverProps } = useHover({
        onHoverChange: (isHovering) => {
            //console.log(isHovering, link, link.children);

            //link.traverse((x) => console.log(x.urdfName, x.type, x.name));
            //console.log(c.urdfName, c.type, c.name, c.isURDFLink);

            const c = model;
            if (c.type === 'Mesh') {
                if (isHovering) {
                    console.log(c, c.type, c.parent.type);
                }

                const mesh = c as THREE.Mesh;
                if (isHovering === false && original.current !== null) {
                    mesh.material = original.current;
                } else {
                    original.current = mesh.material;
                    mesh.material = new THREE.MeshPhongMaterial({
                        shininess: 10,
                        color: '#ff00ff', //this.highlightColor,
                        emissive: '#ff00ff', //this.highlightColor,
                        emissiveIntensity: 0.25,
                    });
                }
            }
        },
    });

    const [meshMaterialMap, setMeshMaterialMap] = useState<Map<string, THREE.Material>>(new Map());

    useEffect(() => {
        const newMap = new Map();
        model.traverse((mesh) => {
            if (mesh instanceof THREE.Mesh) {
                console.log(mesh, mesh.name, mesh.id, mesh.uuid);

                newMap.set(mesh.uuid, mesh.material);
                mesh.addEventListener('onpointerenter', (event) => {
                    console.log('enter', event);
                });
                mesh.addEventListener('onpointerleave', (event) => {
                    console.log('leave', event);
                });
            }
        });
        setMeshMaterialMap(newMap);
    }, [model]);

    return (
        <View>
            <div {...hoverProps}>
                <View backgroundColor={'red-400'}>
                    <Text>
                        {model.name} : {model.type}
                    </Text>
                </View>
            </div>
            {model.children.length > 0 && (
                <ul>
                    <Flex gap='size-50' direction={'column'} marginStart='size-100'>
                        {model.children.map((child) => {
                            return (
                                <li>
                                    <ModelTree model={child} />
                                </li>
                            );
                        })}
                    </Flex>
                </ul>
            )}
        </View>
    );
};

const ModelControl = ({ model }: { model: URDFRobot }) => {
    console.log({ model, links: model.links, joints: model.joints, visual: model.visual });
    const [showModel, setShowModel] = useState(false);

    return (
        <Flex direction='column'>
            <Heading level={4}>{model.uuid}</Heading>
            <JointControls model={model} />

            <ul>
                <Flex gap='size-50' direction={'column'}>
                    {Object.keys(model.links).map((linkKey) => (
                        <li>
                            <LinkControl name={linkKey} link={model.links[linkKey]} />
                        </li>
                    ))}
                </Flex>
            </ul>
            {showModel === false ? (
                <Button onPress={() => setShowModel(true)}>Show</Button>
            ) : (
                <View height='200px' overflow='scorll'>
                    <ModelTree model={model} />
                </View>
            )}
        </Flex>
    );
};

export const RobotControls = () => {
    const { models } = useAction();

    return (
        <View>
            {models.map((model) => (
                <ModelControl key={model.uuid} model={model} />
            ))}
        </View>
    );
};
