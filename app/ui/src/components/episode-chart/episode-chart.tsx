import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

function buildChartData(actions: number[][], joints: string[], fps: number) {
  return actions.map((row, idx) => ({
    index: idx / fps,              
    ...joints.reduce((acc, joint, index) => ({...acc, [joint]: row[index]}),{})
  }));
}

const lineColors = ["#ff7300", "#387908", "#8884d8", "#82ca9d", "#ffbb28", "#a83279"];
interface EpisodeChartProps {
    actions: number[][];
    joints: string[];
    fps: number;
}

function toCapitalizedWords(name: string) {
    var words = name.match(/[A-Za-z][a-z]*/g) || [];

    return words.map(capitalize).join(" ");
}

function capitalize(word:  string) {
    return word.charAt(0).toUpperCase() + word.substring(1);
}

export default function EpisodeChart({ actions, joints, fps}: EpisodeChartProps) {
    const chartData = buildChartData(actions, joints, fps);

  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart data={chartData} margin={{ top: 20, right: 20, left: 20, bottom: 20 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="index" label={{ value: "Index", position: "insideBottomRight", offset: -10 }} />
        <YAxis label={{ value: "Value (deg)", angle: -90, position: "insideLeft" }} />

        <Tooltip />
        <Legend verticalAlign="top" height={36} />

        {joints.map((joint, i) => (
          <Line
            key={joint}
            type="monotone"
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