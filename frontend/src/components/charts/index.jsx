import React from 'react'
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  AreaChart,
  Area,
} from 'recharts'
import { motion } from 'framer-motion'

const COLORS = ['#7c00ff', '#3b82f6', '#10b981', '#f59e0b', '#ec4899', '#8b5cf6']

const chartVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.5 } },
}

// Custom Premium Tooltip Component
const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-slate-900/90 dark:bg-slate-950/95 backdrop-blur-xl border border-slate-800/80 p-3 rounded-2xl shadow-glow-purple text-xs text-white">
        <p className="font-extrabold text-slate-200 mb-1.5 border-b border-slate-800 pb-1">{label}</p>
        <div className="space-y-1">
          {payload.map((item, idx) => (
            <p key={idx} className="font-bold flex items-center gap-2">
              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: item.color || item.fill || '#7c00ff' }} />
              <span className="text-slate-400 font-medium">{item.name}:</span>
              <span className="text-white font-extrabold">{item.value?.toLocaleString()}</span>
            </p>
          ))}
        </div>
      </div>
    )
  }
  return null
}

export const SimpleBarChart = ({ data, title, dataKey, xAxisKey }) => {
  return (
    <motion.div variants={chartVariants} className="glass shadow-card hover:shadow-glass p-6 transition-all duration-300">
      <h3 className="text-lg font-extrabold text-slate-900 dark:text-white mb-6 font-heading">{title}</h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
          <defs>
            <linearGradient id="barBlueGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#7c00ff" stopOpacity={1} />
              <stop offset="100%" stopColor="#3b82f6" stopOpacity={0.4} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.12)" />
          <XAxis 
            dataKey={xAxisKey} 
            tick={{ fill: '#64748b', fontSize: 10, fontWeight: 600 }}
            axisLine={{ stroke: 'rgba(148, 163, 184, 0.2)' }}
            tickLine={false}
          />
          <YAxis 
            tick={{ fill: '#64748b', fontSize: 10, fontWeight: 600 }}
            axisLine={{ stroke: 'rgba(148, 163, 184, 0.2)' }}
            tickLine={false}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(148, 163, 184, 0.05)' }} />
          <Bar dataKey={dataKey} fill="url(#barBlueGradient)" radius={[6, 6, 0, 0]} maxBarSize={45} />
        </BarChart>
      </ResponsiveContainer>
    </motion.div>
  )
}

export const SimpleLineChart = ({ data, title, dataKey, xAxisKey }) => {
  return (
    <motion.div variants={chartVariants} className="glass shadow-card hover:shadow-glass p-6 transition-all duration-300">
      <h3 className="text-lg font-extrabold text-slate-900 dark:text-white mb-6 font-heading">{title}</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.12)" />
          <XAxis 
            dataKey={xAxisKey} 
            tick={{ fill: '#64748b', fontSize: 10, fontWeight: 600 }}
            axisLine={{ stroke: 'rgba(148, 163, 184, 0.2)' }}
            tickLine={false}
          />
          <YAxis 
            tick={{ fill: '#64748b', fontSize: 10, fontWeight: 600 }}
            axisLine={{ stroke: 'rgba(148, 163, 184, 0.2)' }}
            tickLine={false}
          />
          <Tooltip content={<CustomTooltip />} />
          <Line
            type="monotone"
            dataKey={dataKey}
            stroke="#7c00ff"
            strokeWidth={3}
            dot={{ fill: '#7c00ff', r: 4, strokeWidth: 2, stroke: '#fff' }}
            activeDot={{ r: 6, fill: '#3b82f6', stroke: '#fff', strokeWidth: 2 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </motion.div>
  )
}

export const SimplePieChart = ({ data, title, dataKey }) => {
  return (
    <motion.div variants={chartVariants} className="glass shadow-card hover:shadow-glass p-6 transition-all duration-300">
      <h3 className="text-lg font-extrabold text-slate-900 dark:text-white mb-6 font-heading">{title}</h3>
      <ResponsiveContainer width="100%" height={300}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            labelLine={false}
            label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`}
            outerRadius={95}
            innerRadius={45}
            fill="#8884d8"
            dataKey={dataKey}
            paddingAngle={3}
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip content={<CustomTooltip />} />
        </PieChart>
      </ResponsiveContainer>
    </motion.div>
  )
}

export const AreaChartComponent = ({ data, title, dataKey, xAxisKey }) => {
  return (
    <motion.div variants={chartVariants} className="glass shadow-card hover:shadow-glass p-6 transition-all duration-300">
      <h3 className="text-lg font-extrabold text-slate-900 dark:text-white mb-6 font-heading">{title}</h3>
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={data} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
          <defs>
            <linearGradient id="areaColorGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.4} />
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.12)" />
          <XAxis 
            dataKey={xAxisKey} 
            tick={{ fill: '#64748b', fontSize: 10, fontWeight: 600 }}
            axisLine={{ stroke: 'rgba(148, 163, 184, 0.2)' }}
            tickLine={false}
          />
          <YAxis 
            tick={{ fill: '#64748b', fontSize: 10, fontWeight: 600 }}
            axisLine={{ stroke: 'rgba(148, 163, 184, 0.2)' }}
            tickLine={false}
          />
          <Tooltip content={<CustomTooltip />} />
          <Area
            type="monotone"
            dataKey={dataKey}
            stroke="#3b82f6"
            strokeWidth={3}
            fillOpacity={1}
            fill="url(#areaColorGradient)"
          />
        </AreaChart>
      </ResponsiveContainer>
    </motion.div>
  )
}

export const RadarChartComponent = ({ data, title, dataKey }) => {
  return (
    <motion.div variants={chartVariants} className="glass shadow-card hover:shadow-glass p-6 transition-all duration-300">
      <h3 className="text-lg font-extrabold text-slate-900 dark:text-white mb-6 font-heading">{title}</h3>
      <ResponsiveContainer width="100%" height={300}>
        <RadarChart data={data}>
          <PolarGrid stroke="rgba(148, 163, 184, 0.12)" />
          <PolarAngleAxis dataKey="name" tick={{ fill: '#64748b', fontSize: 10, fontWeight: 600 }} />
          <PolarRadiusAxis tick={{ fill: '#64748b', fontSize: 10 }} />
          <Radar 
            name="Allocation Value" 
            dataKey={dataKey} 
            stroke="#7c00ff" 
            fill="#7c00ff" 
            fillOpacity={0.3} 
            strokeWidth={2}
          />
          <Tooltip content={<CustomTooltip />} />
        </RadarChart>
      </ResponsiveContainer>
    </motion.div>
  )
}

export const MultiSeriesBarChart = ({ data, title, series, xAxisKey }) => {
  return (
    <motion.div variants={chartVariants} className="glass shadow-card hover:shadow-glass p-6 transition-all duration-300">
      <h3 className="text-lg font-extrabold text-slate-900 dark:text-white mb-6 font-heading">{title}</h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.12)" />
          <XAxis 
            dataKey={xAxisKey} 
            tick={{ fill: '#64748b', fontSize: 10, fontWeight: 600 }}
            axisLine={{ stroke: 'rgba(148, 163, 184, 0.2)' }}
            tickLine={false}
          />
          <YAxis 
            tick={{ fill: '#64748b', fontSize: 10, fontWeight: 600 }}
            axisLine={{ stroke: 'rgba(148, 163, 184, 0.2)' }}
            tickLine={false}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(148, 163, 184, 0.05)' }} />
          <Legend tick={{ fill: '#64748b', fontSize: 11, fontWeight: 600 }} wrapperStyle={{ paddingTop: '15px' }} />
          {series.map((item, idx) => (
            <Bar 
              key={item.key} 
              dataKey={item.key} 
              fill={COLORS[idx % COLORS.length]} 
              radius={[6, 6, 0, 0]} 
              maxBarSize={30}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </motion.div>
  )
}
