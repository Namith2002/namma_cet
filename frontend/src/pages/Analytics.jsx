import React from 'react'
import { motion } from 'framer-motion'
import { useQuery } from '@tanstack/react-query'
import { Loader, useToast, ToastContainer } from '../components/common'
import { StatCard } from '../components/cards'
import {
  SimpleBarChart,
  SimplePieChart,
  AreaChartComponent,
  MultiSeriesBarChart,
} from '../components/charts'
import { analyticsService } from '../services/endpoints'
import { FiBarChart2, FiTrendingUp, FiBook, FiUsers } from 'react-icons/fi'

const Analytics = () => {
  const { toasts, removeToast, error: showError } = useToast()

  const { data: analytics, isLoading } = useQuery({
    queryKey: ['analytics'],
    queryFn: () => analyticsService.getAnalytics(),
    onError: (err) => showError(err.message),
  })

  const { data: coursePopularity, isLoading: courseLoading } = useQuery({
    queryKey: ['coursePopularity'],
    queryFn: () => analyticsService.getCoursPopularity(),
    onError: (err) => showError(err.message),
  })

  const { data: cutoffDist, isLoading: cutoffLoading } = useQuery({
    queryKey: ['cutoffDistribution'],
    queryFn: () => analyticsService.getCutoffDistribution(),
    onError: (err) => showError(err.message),
  })

  const { data: categoryAnalysis, isLoading: categoryLoading } = useQuery({
    queryKey: ['categoryAnalysis'],
    queryFn: () => analyticsService.getCategoryAnalysis(),
    onError: (err) => showError(err.message),
  })

  if (isLoading || courseLoading || cutoffLoading || categoryLoading) {
    return <Loader fullScreen />
  }

  const statsCards = [
    {
      title: 'Total Colleges',
      value: analytics?.total_colleges || '37',
      icon: FiUsers,
      color: 'blue',
      progress: 78,
      trend: { value: 4.8, positive: true },
    },
    {
      title: 'Total Courses',
      value: analytics?.total_courses || '98',
      icon: FiBook,
      color: 'purple',
      progress: 65,
      trend: { value: 2.1, positive: true },
    },
    {
      title: 'Average Cutoff',
      value: analytics?.average_cutoff?.toLocaleString() || '5,420',
      icon: FiTrendingUp,
      color: 'green',
      progress: 82,
      trend: { value: 1.5, positive: true },
    },
    {
      title: 'Total Categories',
      value: analytics?.total_categories || '24',
      icon: FiBarChart2,
      color: 'orange',
      progress: 50,
      trend: { value: 0, positive: true },
    },
  ]

  return (
    <div className="min-h-screen py-12 relative overflow-hidden">
      {/* Background radial glows */}
      <div className="absolute top-10 left-10 w-[500px] h-[500px] bg-primary-500/10 dark:bg-primary-500/5 rounded-full blur-[130px] pointer-events-none" />
      <div className="absolute bottom-10 right-10 w-[500px] h-[500px] bg-accent-blue/10 dark:bg-accent-blue/5 rounded-full blur-[130px] pointer-events-none" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-12 text-center md:text-left"
        >
          <h1 className="text-4xl md:text-5xl font-extrabold mb-4">
            Analytics <span className="gradient-text">Dashboard</span>
          </h1>
          <p className="text-lg text-slate-600 dark:text-slate-400 max-w-2xl">
            Gain deep insights and explore trends about engineering and medical admissions, cutoffs, and course demand.
          </p>
        </motion.div>

        {/* Stats Cards Grid */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-12"
        >
          {statsCards.map((stat, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.05 }}
            >
              <StatCard {...stat} />
            </motion.div>
          ))}
        </motion.div>

        {/* Charts Grid */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="grid lg:grid-cols-2 gap-6 mb-12"
        >
          {coursePopularity && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.05 }}
            >
              <SimpleBarChart
                data={coursePopularity.slice(0, 10)}
                title="Top 10 Popular Courses By Intake"
                dataKey="popularity"
                xAxisKey="name"
              />
            </motion.div>
          )}

          {cutoffDist && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
            >
              <SimplePieChart
                data={cutoffDist}
                title="Colleges Cutoff Range Distribution"
                dataKey="count"
              />
            </motion.div>
          )}

          {categoryAnalysis && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.15 }}
            >
              <MultiSeriesBarChart
                data={categoryAnalysis}
                title="Category-wise Seat Allocations"
                series={[{ key: 'count' }]}
                xAxisKey="category"
              />
            </motion.div>
          )}

          {coursePopularity && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
            >
              <AreaChartComponent
                data={coursePopularity.slice(0, 8)}
                title="Course Popularity Demand Index"
                dataKey="popularity"
                xAxisKey="name"
              />
            </motion.div>
          )}
        </motion.div>
      </div>

      <ToastContainer toasts={toasts} removeToast={removeToast} />
    </div>
  )
}

export default Analytics
