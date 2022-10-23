import React, { PureComponent } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const data = [
  {
    name: '월',
    과잉행동: 4000,
    반복행동: 2400,
    행동전환성: 2400,
  },
  {
    name: '화',
    과잉행동: 3000,
    반복행동: 1398,
    행동전환성: 2210,
  },
  {
    name: '수',
    과잉행동: 2000,
    반복행동: 9800,
    행동전환성: 2290,
  },
  {
    name: '목',
    과잉행동: 2780,
    반복행동: 3908,
    행동전환성: 2000,
  },
  {
    name: '금',
    과잉행동: 1890,
    반복행동: 4800,
    행동전환성: 2181,
  },
  {
    name: '토',
    과잉행동: 2390,
    반복행동: 3800,
    행동전환성: 2500,
  },
  {
    name: '일',
    과잉행동: 3490,
    반복행동: 4300,
    행동전환성: 2100,
  },
];

export default class Charts extends PureComponent {

  render() {
    return (
      <ResponsiveContainer width="66%" height="100%">
        <BarChart
          width={500}
          height={500}
          data={data}
          margin={{
            top: 20,
            right: 30,
            left: 20,
            bottom: 5,
          }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Bar dataKey="반복행동" stackId="a" fill="#8884d8" />
          <Bar dataKey="행동전환성" stackId="a" fill="#82ca9d" />
          <Bar dataKey="과잉행동" stackId="a" fill="#2347c1" />
        </BarChart>
      </ResponsiveContainer>
    );
  }
}