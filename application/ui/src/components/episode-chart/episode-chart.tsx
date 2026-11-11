import { useCallback, useEffect, useMemo, useRef } from 'react';

import uPlot from 'uplot';

import { resolveCssColor, useUPlot } from '../uplot/use-uplot';

import classes from './episode-chart.module.css';

const lineColors = ['#ff7300', '#387908', '#8884d8', '#82ca9d', '#ffbb28', '#a83279', '#aaffff'];

const PLAYHEAD_COLOR = '#ffffff';
const PLOT_HEIGHT = 264;
const TIME_TICK_INCREMENTS = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300];

function toCapitalizedWords(name: string) {
    const words = name.match(/[A-Za-z][a-z]*/g) || [];

    return words.map(capitalize).join(' ');
}

function capitalize(word: string) {
    return word.charAt(0).toUpperCase() + word.substring(1);
}

function isPosJoint(name: string) {
    return name.endsWith('.pos');
}

function stripPosSuffix(name: string) {
    return name.replace(/\.pos$/, '');
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
    const posJoints = useMemo(
        () => joints.flatMap((joint, column) => (isPosJoint(joint) ? [{ joint, column }] : [])),
        [joints]
    );

    const chartData = useMemo<uPlot.AlignedData>(
        () => [
            actions.map((_, frame) => frame / fps),
            ...posJoints.map(({ column }) => actions.map((row) => row[column])),
        ],
        [actions, posJoints, fps]
    );

    // The plot is built once per set of joints; these refs let its hooks and listeners read the
    // latest props without rebuilding it on every playhead tick.
    const timeRef = useRef(time);
    const playerRef = useRef({ seek, play, pause, isPlaying });

    useEffect(() => {
        playerRef.current = { seek, play, pause, isPlaying };
    }, [seek, play, pause, isPlaying]);

    const buildOptions = useCallback(
        (container: HTMLDivElement): Omit<uPlot.Options, 'width' | 'height'> => {
            const axisColor = resolveCssColor(container, 'var(--spectrum-global-color-gray-700)');
            const gridColor = resolveCssColor(container, 'var(--spectrum-global-color-gray-300)');

            const axis = {
                stroke: axisColor,
                grid: { stroke: gridColor, width: 1 },
                ticks: { stroke: gridColor, width: 1 },
            };

            const drawPlayhead = (plot: uPlot) => {
                const { ctx, bbox } = plot;
                const x = plot.valToPos(timeRef.current, 'x', true);

                ctx.save();
                ctx.strokeStyle = PLAYHEAD_COLOR;
                ctx.lineWidth = 2 * uPlot.pxRatio;
                ctx.beginPath();
                ctx.moveTo(x, bbox.top);
                ctx.lineTo(x, bbox.top + bbox.height);
                ctx.stroke();
                ctx.restore();
            };

            const attachSeeking = (plot: uPlot) => {
                let resumeOnRelease = false;

                const timeAt = (event: MouseEvent) => {
                    const rect = plot.over.getBoundingClientRect();
                    const left = Math.min(Math.max(event.clientX - rect.left, 0), rect.width);

                    return plot.posToVal(left, 'x');
                };

                const onMove = (event: MouseEvent) => playerRef.current.seek(timeAt(event));

                const onUp = (event: MouseEvent) => {
                    window.removeEventListener('mousemove', onMove);
                    playerRef.current.seek(timeAt(event));
                    if (resumeOnRelease) playerRef.current.play();
                };

                plot.over.addEventListener('mousedown', () => {
                    resumeOnRelease = playerRef.current.isPlaying;
                    playerRef.current.pause();
                    window.addEventListener('mousemove', onMove);
                    window.addEventListener('mouseup', onUp, { once: true });
                });
            };

            return {
                padding: [20, 20, 0, 0],
                legend: { live: false },
                cursor: {
                    y: false,
                    points: { show: false },
                    drag: { x: false, y: false },
                },
                scales: {
                    x: { time: false, range: (_plot, _min, max) => [0, max] },
                },
                axes: [
                    { ...axis, label: 'Time (s)', incrs: TIME_TICK_INCREMENTS },
                    { ...axis, label: 'Value (deg)' },
                ],
                series: [
                    {},
                    ...posJoints.map(({ joint }, index) => ({
                        label: toCapitalizedWords(stripPosSuffix(joint)),
                        stroke: lineColors[index % lineColors.length],
                        width: 2,
                        points: { show: false },
                    })),
                ],
                hooks: {
                    init: [attachSeeking],
                    draw: [drawPlayhead],
                },
            };
        },
        [posJoints]
    );

    const { containerRef, plotRef } = useUPlot(buildOptions, chartData, PLOT_HEIGHT);

    useEffect(() => {
        timeRef.current = time;
        // Repaint from cached paths so only the playhead moves.
        plotRef.current?.redraw(false);
    }, [time, plotRef]);

    return <div ref={containerRef} className={classes.chart} />;
}
