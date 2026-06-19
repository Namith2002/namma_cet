import React from 'react'
import { motion } from 'framer-motion'
import { FiArrowRight, FiCheck, FiX } from 'react-icons/fi'
import { Button } from '../common/Button'
import { calculateEligibility, getEligibilityStatus, getEligibilityColor } from '../../utils/helpers'
import { usePrediction } from '../../context/PredictionContext'

export const StatCard = ({ title, value, icon: Icon, trend, progress, color = 'blue' }) => {
  const colorClasses = {
    blue: 'bg-blue-500/10 text-blue-500 border border-blue-500/20',
    green: 'bg-emerald-500/10 text-emerald-500 border border-emerald-500/20',
    purple: 'bg-primary-500/10 text-primary-500 border border-primary-500/20',
    orange: 'bg-amber-500/10 text-amber-500 border border-amber-500/20',
  }

  return (
    <motion.div
      className="glass p-6 relative overflow-hidden group shadow-card hover:shadow-glass hover:-translate-y-1 transition-all duration-300"
      whileHover={{ scale: 1.02 }}
      transition={{ type: 'spring', stiffness: 300 }}
    >
      <div className="flex items-start justify-between relative z-10">
        <div className="flex-1 pr-4">
          <p className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">{title}</p>
          <p className="text-3xl font-extrabold text-slate-900 dark:text-white mt-2 leading-none">{value}</p>
          {trend && (
            <p className={`text-xs mt-3 font-semibold flex items-center gap-1 ${trend.positive ? 'text-emerald-500' : 'text-rose-500'}`}>
              {trend.positive ? '+' : ''}{trend.value}% vs last year
            </p>
          )}
          {progress !== undefined && (
            <div className="w-full mt-4 bg-slate-100 dark:bg-slate-800/80 rounded-full h-1.5 overflow-hidden">
              <motion.div 
                className="bg-gradient-primary h-full"
                initial={{ width: 0 }}
                animate={{ width: `${progress}%` }}
                transition={{ duration: 0.8, ease: 'easeOut' }}
              />
            </div>
          )}
        </div>
        <div className={`p-3 rounded-2xl ${colorClasses[color]} shadow-sm group-hover:scale-110 transition-transform duration-300`}>
          <Icon size={22} />
        </div>
      </div>
      {/* Subtle radial glow inside card on hover */}
      <div className="absolute -inset-px bg-gradient-to-r from-primary-500/10 to-accent-blue/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-2xl pointer-events-none" />
    </motion.div>
  )
}

export const CollegeCard = ({ college, onViewDetails, userRank, category }) => {
  const eligible = calculateEligibility(userRank, college.cutoff_rank)
  const { comparedColleges, toggleCompareCollege } = usePrediction()
  const isCompared = comparedColleges.some((c) => c.id === college.id)
  const isMaxCompareReached = comparedColleges.length >= 3

  return (
    <motion.div
      className={`glass-card h-full flex flex-col justify-between ${
        isCompared ? 'ring-2 ring-primary-500/80 shadow-glow-purple border-transparent' : ''
      }`}
      whileHover={{ scale: 1.02, y: -4 }}
      transition={{ type: 'spring', stiffness: 300, damping: 25 }}
    >
      <div>
        <div className="flex items-start justify-between mb-4">
          <div>
            <h3 className="text-base font-extrabold text-slate-900 dark:text-white leading-tight mb-1">
              {college.name}
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400 font-bold">{college.code}</p>
          </div>
          {eligible !== null && (
            <div
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-bold ${
                eligible
                  ? 'bg-green-100 dark:bg-green-950/30 text-green-700 dark:text-green-300'
                  : 'bg-red-100 dark:bg-red-950/30 text-red-700 dark:text-red-300'
              }`}
            >
              {eligible ? <FiCheck size={12} /> : <FiX size={12} />}
              {getEligibilityStatus(eligible)}
            </div>
          )}
        </div>

        <div className="space-y-2 mb-4 text-xs">
          <div className="flex justify-between items-center">
            <span className="text-slate-500 dark:text-slate-400 font-medium">Allocated Course</span>
            <span className="font-bold text-slate-950 dark:text-slate-100">{college.course}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-slate-500 dark:text-slate-400 font-medium">Cutoff Rank</span>
            <span className="font-extrabold text-slate-950 dark:text-slate-100">{college.cutoff_rank}</span>
          </div>
        </div>
      </div>

      <div className="flex gap-2 mt-4 pt-3 border-t border-slate-200/50 dark:border-slate-800/50">
        <Button
          variant="outline"
          size="sm"
          className="flex-1 rounded-xl text-xs font-bold py-2 border-slate-200 dark:border-slate-800"
          onClick={() => {
            if (onViewDetails) onViewDetails(college.id);
          }}
        >
          Details
        </Button>
        <Button
          variant={isCompared ? 'primary' : 'outline'}
          size="sm"
          className={`flex-1 rounded-xl text-xs font-bold py-2 border-slate-200 dark:border-slate-800 transition-all duration-300 ${
            isCompared 
              ? 'bg-gradient-primary border-none shadow-glow-purple text-white' 
              : 'hover:border-primary-500'
          }`}
          disabled={!isCompared && isMaxCompareReached}
          onClick={() => toggleCompareCollege(college)}
        >
          {isCompared ? 'Added' : 'Compare'}
        </Button>
      </div>
    </motion.div>
  )
}

export const CourseCard = ({ course, onSelect, isSelected = false }) => {
  return (
    <motion.div
      className={`glass-card cursor-pointer ${isSelected ? 'ring-2 ring-blue-500' : ''}`}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      onClick={() => onSelect(course.id)}
      transition={{ type: 'spring', stiffness: 300 }}
    >
      <div className="flex items-start justify-between mb-3">
        <div>
          <h3 className="text-lg font-bold text-gray-900 dark:text-white">{course.name}</h3>
          <p className="text-sm text-gray-600 dark:text-gray-400">{course.code}</p>
        </div>
        {isSelected && (
          <div className="flex items-center justify-center w-6 h-6 bg-blue-500 rounded-full">
            <FiCheck size={16} className="text-white" />
          </div>
        )}
      </div>

      <div className="space-y-2">
        <div className="flex justify-between items-center">
          <span className="text-sm text-gray-600 dark:text-gray-400">Popularity</span>
          <div className="w-24 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-blue-500"
              initial={{ width: 0 }}
              animate={{ width: `${course.popularity}%` }}
              transition={{ duration: 0.5, ease: 'easeOut' }}
            />
          </div>
        </div>
        <div className="flex justify-between items-center text-sm">
          <span className="text-gray-600 dark:text-gray-400">Colleges Offering</span>
          <span className="font-semibold text-gray-900 dark:text-white">
            {course.college_count}
          </span>
        </div>
      </div>
    </motion.div>
  )
}
