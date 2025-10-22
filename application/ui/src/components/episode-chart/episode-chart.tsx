import { useRef, useState } from 'react';

import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, XAxis, YAxis } from 'recharts';
import { CategoricalChartState } from 'recharts/types/chart/types';

function buildChartData(actions: number[][], joints: string[], fps: number) {
    return actions.map((row, idx) => ({
        index: idx / fps,
        ...joints.reduce((acc, joint, index) => ({ ...acc, [joint]: row[index] }), {}),
    }));
}

const lineColors = ['#ff7300', '#387908', '#8884d8', '#82ca9d', '#ffbb28', '#a83279'];

function toCapitalizedWords(name: string) {
    const words = name.match(/[A-Za-z][a-z]*/g) || [];

    return words.map(capitalize).join(' ');
}

function capitalize(word: string) {
    return word.charAt(0).toUpperCase() + word.substring(1);
}

interface EpisodeChartProps {
    actions: number[][];
    joints: string[];
    fps: number;
    time: number;
    seek: (time: number) => void;
    play: () => void;
    pause: () => void;
    isPlaying: boolean;
}

export default function EpisodeChart({ actions, joints, fps, time, seek, play, pause, isPlaying }: EpisodeChartProps) {
    const [hoverPosition, setHoverPosition] = useState<number | undefined>(undefined);
    const [mouseDown, setMouseDown] = useState<boolean>(false);
    const chartData = buildChartData(actions, joints, fps);
    const ticks = [...Array(Math.floor((actions.length / fps) * 2)).keys()].map((m) => m / 2);

    const continuePlayingOnRelease = useRef<boolean>(false);

    const handleMouseMove = (nextState: CategoricalChartState) => {
        if (nextState.activeLabel === undefined) {
            setHoverPosition(undefined);
        } else {
            const newTime = parseFloat(nextState.activeLabel ?? '0');
            if (mouseDown) {
                seek(newTime);
            }
            setHoverPosition(newTime);
        }
    };

    const handleMouseUp = (nextState: CategoricalChartState) => {
        if (nextState.activeLabel !== undefined) {
            const newTime = parseFloat(nextState.activeLabel ?? '0');
            seek(newTime);
        }
        setMouseDown(false);
        if (continuePlayingOnRelease.current) {
            play();
        }
    };

    const handleMouseDown = () => {
        setMouseDown(true);
        continuePlayingOnRelease.current = isPlaying;
        pause();
    };

    return (
        <ResponsiveContainer width='100%' height={300} style={{ userSelect: 'none' }}>
            <LineChart
                data={chartData}
                margin={{ top: 20, right: 20, left: 20, bottom: 20 }}
                onMouseUp={handleMouseUp}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
            >
                <CartesianGrid opacity={0.2} />
                <XAxis
                    dataKey='index'
                    label={{ value: 'Time (s)', position: 'insideBottomRight', offset: -10 }}
                    ticks={ticks}
                    type='number'
                />
                <YAxis label={{ value: 'Value (deg)', angle: -90, position: 'insideLeft' }} />

                <Legend verticalAlign='bottom' height={36} />
                {hoverPosition !== undefined && (
                    <ReferenceLine x={hoverPosition} stroke='#5ac3f8' label='' strokeWidth={2} />
                )}
                <ReferenceLine x={time} stroke='#ffffff' label='' strokeWidth={2} />

                {joints.map((joint, i) => (
                    <Line
                        isAnimationActive={false}
                        key={joint}
                        type='monotone'
                        dataKey={joint}
                        stroke={lineColors[i]}
                        name={toCapitalizedWords(joint)}
                        dot={false}
                        strokeWidth={2}
                    />
                ))}
            </LineChart>
        </ResponsiveContainer>
    );
}
