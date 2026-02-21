import { useEffect, useRef } from 'react';

import * as THREE from 'three';
import { URDFRobot } from 'urdf-loader';

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/** Semantic highlight colors. */
export type HighlightColor = 'accent' | 'positive' | 'negative';

/** A single joint highlight entry: which joint and what color. */
export interface JointHighlight {
    joint: string;
    color: HighlightColor;
}

// ---------------------------------------------------------------------------
// Emissive color mapping
// ---------------------------------------------------------------------------

const EMISSIVE_MAP: Record<HighlightColor, THREE.Color> = {
    accent: new THREE.Color(0x00c7fd),   // cyan — matches the 3D grid
    positive: new THREE.Color(0x2dc937), // green — motor OK
    negative: new THREE.Color(0xd32f2f), // red — motor missing
};

const HIGHLIGHT_EMISSIVE_INTENSITY = 0.5;

/** Material name used by the STS3215 servo motor meshes in the URDF */
const MOTOR_MATERIAL_NAME = 'sts3215';

interface SavedMaterial {
    mesh: THREE.Mesh;
    original: THREE.Material;
}

/**
 * Collect all Mesh descendants of a node, stopping traversal at child
 * URDFJoints so only the "direct" link geometry is returned (not
 * downstream segments of the kinematic chain).
 */
function collectLinkMeshes(root: THREE.Object3D): THREE.Mesh[] {
    const meshes: THREE.Mesh[] = [];

    function walk(node: THREE.Object3D) {
        // Stop at sub-joints (but not the root itself)
        if (node !== root && (node as any).isURDFJoint) return;
        if ((node as THREE.Mesh).isMesh) meshes.push(node as THREE.Mesh);
        for (const child of node.children) walk(child);
    }

    walk(root);
    return meshes;
}

/**
 * Collect motor meshes (sts3215 material) from a joint's parent link.
 * Returns all meshes from the parent link filtered to motor material,
 * falling back to all meshes if no motor material is found.
 */
function collectMotorMeshesForJoint(robot: URDFRobot, jointName: string): THREE.Mesh[] {
    const joint = robot.joints[jointName];
    if (!joint) return [];

    // The motor lives in the joint's parent link (the upstream side)
    const parentLink = joint.parent;
    if (!parentLink || !(parentLink as any).isURDFLink) return [];

    const allMeshes = collectLinkMeshes(parentLink);

    // Filter to only motor meshes (material name contains "sts3215")
    const motorMeshes = allMeshes.filter((mesh) => {
        const mat = mesh.material as THREE.Material;
        return mat.name.toLowerCase().includes(MOTOR_MATERIAL_NAME);
    });

    return motorMeshes.length > 0 ? motorMeshes : allMeshes;
}

/**
 * Highlights the STS3215 motor meshes belonging to one or more joints'
 * **parent** links.
 *
 * In the SO101 URDF, the motor that drives each joint is physically located
 * in the joint's parent link (the upstream side). The child link is the part
 * the motor *moves*. We filter to only meshes whose material name matches
 * "sts3215" so only the motor is highlighted, not the 3D-printed structural
 * parts of the parent link.
 *
 * Accepts an array of `JointHighlight` entries, each specifying a joint name
 * and a semantic color (`'accent'`, `'positive'`, or `'negative'`). Pass an
 * empty array to clear highlights. When the value changes, the previous
 * highlight is restored and the new joints are highlighted.
 *
 * Materials are cloned so shared material instances are never mutated.
 */
export function useJointHighlight(
    robot: URDFRobot | undefined,
    highlights: JointHighlight[],
) {
    const saved = useRef<SavedMaterial[]>([]);

    // Stable string key for the dependency array
    const highlightsKey = highlights.map((h) => `${h.joint}:${h.color}`).join(',');

    useEffect(() => {
        // 1. Restore any previously highlighted meshes
        for (const s of saved.current) {
            s.mesh.material = s.original;
        }
        saved.current = [];

        // 2. Nothing to highlight
        if (!robot || highlights.length === 0) return;

        // 3. Collect motor meshes per joint with associated color (deduplicate)
        const seen = new Set<THREE.Mesh>();
        const entries: { mesh: THREE.Mesh; emissive: THREE.Color }[] = [];
        for (const { joint, color } of highlights) {
            const emissive = EMISSIVE_MAP[color];
            for (const mesh of collectMotorMeshesForJoint(robot, joint)) {
                if (!seen.has(mesh)) {
                    seen.add(mesh);
                    entries.push({ mesh, emissive });
                }
            }
        }

        for (const { mesh, emissive } of entries) {
            // Save original material reference
            saved.current.push({ mesh, original: mesh.material as THREE.Material });

            // Clone so we don't mutate shared materials
            const highlighted = (mesh.material as THREE.Material).clone() as THREE.MeshPhongMaterial;

            if ('emissive' in highlighted) {
                highlighted.emissive.copy(emissive);
                highlighted.emissiveIntensity = HIGHLIGHT_EMISSIVE_INTENSITY;
            } else {
                // Fallback for materials without emissive support
                (highlighted as any).color = emissive.clone();
            }

            mesh.material = highlighted;
        }

        // 4. Cleanup on unmount or before next effect run
        return () => {
            for (const s of saved.current) {
                s.mesh.material = s.original;
            }
            saved.current = [];
        };
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [robot, highlightsKey]);
}
