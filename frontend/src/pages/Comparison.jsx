import React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useQueries } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { FiTrash2, FiArrowLeft, FiPlus, FiCpu, FiCheck, FiX, FiAward, FiBookOpen } from 'react-icons/fi'
import { usePrediction } from '../context/PredictionContext'
import { collegeService } from '../services/endpoints'
import { Loader } from '../components/common'

const Comparison = () => {
  const navigate = useNavigate()
  const { comparedColleges, toggleCompareCollege, clearComparedColleges } = usePrediction()

  // Fetch details for all compared colleges
  const collegeQueries = useQueries({
    queries: comparedColleges.map((c) => ({
      queryKey: ['collegeDetails', c.code || c.id],
      queryFn: () => collegeService.getCollegeDetails(c.code || c.id),
      enabled: !!(c.code || c.id),
    })),
  })

  const isLoading = collegeQueries.some((q) => q.isLoading)
  const isError = collegeQueries.some((q) => q.isError)

  const loadedColleges = collegeQueries
    .map((q) => q.data)
    .filter(Boolean)

  // Empty state
  if (comparedColleges.length === 0) {
    return (
      <div className="min-h-screen py-16 flex flex-col justify-center items-center relative overflow-hidden">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary-500/10 rounded-full blur-[120px] pointer-events-none" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-accent-blue/10 rounded-full blur-[120px] pointer-events-none" />

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="max-w-md text-center p-8 glass shadow-glow-purple mx-4 relative z-10"
        >
          <div className="w-16 h-16 bg-gradient-primary rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-glow-purple">
            <FiCpu className="text-white text-3xl animate-pulse" />
          </div>
          <h2 className="text-2xl font-extrabold text-slate-900 dark:text-white mb-3 font-heading">
            Comparison Cart Empty
          </h2>
          <p className="text-slate-600 dark:text-slate-400 mb-8 text-sm">
            Select up to 3 colleges from the College Explorer or allocation lists to compare their cutoffs, stats, and courses side-by-side.
          </p>
          <button
            onClick={() => navigate('/colleges')}
            className="w-full py-3 bg-gradient-primary text-white font-extrabold rounded-xl shadow-glow-purple hover:scale-[1.02] active:scale-[0.98] transition-all flex items-center justify-center gap-2"
          >
            <FiPlus size={16} /> Explore Colleges
          </button>
        </motion.div>
      </div>
    )
  }

  if (isLoading) {
    return <Loader fullScreen />
  }

  // Extract all unique courses offered across all compared colleges
  const uniqueCourses = []
  const courseCodes = new Set()
  loadedColleges.forEach((college) => {
    if (Array.isArray(college.courses)) {
      college.courses.forEach((course) => {
        if (!courseCodes.has(course.code)) {
          courseCodes.add(course.code)
          uniqueCourses.push({ code: course.code, name: course.name })
        }
      })
    }
  })

  // Highlight criteria helpers
  const minRankValues = loadedColleges.map((c) => c.min_rank || 0).filter(Boolean)
  const bestMinRank = minRankValues.length > 0 ? Math.min(...minRankValues) : null

  const avgRankValues = loadedColleges.map((c) => c.avg_rank || 0).filter(Boolean)
  const bestAvgRank = avgRankValues.length > 0 ? Math.min(...avgRankValues) : null

  const maxCourseCount = Math.max(...loadedColleges.map((c) => c.course_count || 0))

  return (
    <div className="min-h-screen py-12 relative overflow-hidden">
      {/* Decorative radial gradients */}
      <div className="absolute top-0 right-1/4 w-[500px] h-[500px] bg-primary-500/10 dark:bg-primary-500/5 rounded-full blur-[150px] pointer-events-none" />
      <div className="absolute bottom-0 left-1/4 w-[500px] h-[500px] bg-accent-blue/10 dark:bg-accent-blue/5 rounded-full blur-[150px] pointer-events-none" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        {/* Navigation & Header */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-10">
          <button
            onClick={() => navigate(-1)}
            className="flex items-center gap-2 text-slate-500 hover:text-slate-800 dark:hover:text-white font-bold transition-colors w-fit"
          >
            <FiArrowLeft size={16} /> Back
          </button>
          
          <div className="flex items-center gap-3">
            <button
              onClick={clearComparedColleges}
              className="px-4 py-2 border border-slate-200 dark:border-slate-800 rounded-xl text-xs font-bold text-slate-600 dark:text-slate-400 hover:text-red-500 dark:hover:text-red-400 transition-colors"
            >
              Clear Comparison
            </button>
            <button
              onClick={() => navigate('/colleges')}
              className="px-4 py-2 bg-gradient-primary rounded-xl text-xs font-extrabold text-white shadow-glow-purple hover:scale-[1.02] transition-all flex items-center gap-1.5"
            >
              <FiPlus size={14} /> Add College ({comparedColleges.length}/3)
            </button>
          </div>
        </div>

        <div className="mb-10 text-center md:text-left">
          <h1 className="text-4xl md:text-5xl font-extrabold mb-3">
            College <span className="gradient-text">Comparison</span>
          </h1>
          <p className="text-slate-600 dark:text-slate-400 max-w-xl">
            Analyze cutoff ranks, stats, and course availability side-by-side to discover the most suited institution.
          </p>
        </div>

        {/* Overview cards side-by-side */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
          <AnimatePresence mode="popLayout">
            {loadedColleges.map((college) => {
              const isBestMin = college.min_rank === bestMinRank
              const isBestAvg = college.avg_rank === bestAvgRank
              const isBestCourseCount = college.course_count === maxCourseCount

              return (
                <motion.div
                  key={college.code}
                  layout
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  className={`glass p-6 flex flex-col justify-between relative overflow-hidden ${
                    isBestAvg ? 'ring-2 ring-emerald-500/50 shadow-neon-green border-transparent' : ''
                  }`}
                >
                  {isBestAvg && (
                    <div className="absolute top-4 right-4 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 text-[10px] font-extrabold px-2.5 py-1 rounded-full flex items-center gap-1 border border-emerald-500/30">
                      <FiAward size={10} /> Top Pick
                    </div>
                  )}

                  <div>
                    <div className="mb-4">
                      <span className="text-[10px] bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 font-extrabold px-2 py-0.5 rounded">
                        {college.code}
                      </span>
                      <h3 className="text-lg font-extrabold mt-2 leading-snug text-slate-900 dark:text-white">
                        {college.college_name || college.name}
                      </h3>
                    </div>

                    <div className="space-y-3 my-6">
                      <div className="flex justify-between items-center text-xs py-1 border-b border-slate-100 dark:border-slate-800/50">
                        <span className="text-slate-500 dark:text-slate-400 font-medium">Min Cutoff Rank</span>
                        <span className={`font-extrabold ${isBestMin ? 'text-emerald-500' : 'text-slate-900 dark:text-slate-100'}`}>
                          {college.min_rank?.toLocaleString() || 'N/A'}
                        </span>
                      </div>
                      <div className="flex justify-between items-center text-xs py-1 border-b border-slate-100 dark:border-slate-800/50">
                        <span className="text-slate-500 dark:text-slate-400 font-medium">Avg Cutoff Rank</span>
                        <span className={`font-extrabold ${isBestAvg ? 'text-emerald-500' : 'text-slate-900 dark:text-slate-100'}`}>
                          {Math.round(college.avg_rank)?.toLocaleString() || 'N/A'}
                        </span>
                      </div>
                      <div className="flex justify-between items-center text-xs py-1 border-b border-slate-100 dark:border-slate-800/50">
                        <span className="text-slate-500 dark:text-slate-400 font-medium">Max Cutoff Rank</span>
                        <span className="font-extrabold text-slate-900 dark:text-slate-100">
                          {college.max_rank?.toLocaleString() || 'N/A'}
                        </span>
                      </div>
                      <div className="flex justify-between items-center text-xs py-1 border-b border-slate-100 dark:border-slate-800/50">
                        <span className="text-slate-500 dark:text-slate-400 font-medium">Courses Offered</span>
                        <span className={`font-extrabold ${isBestCourseCount ? 'text-emerald-500' : 'text-slate-900 dark:text-slate-100'}`}>
                          {college.course_count || 'N/A'}
                        </span>
                      </div>
                    </div>
                  </div>

                  <button
                    onClick={() => toggleCompareCollege(college)}
                    className="w-full mt-4 py-2 border border-slate-200 dark:border-slate-800 hover:bg-red-500/10 hover:text-red-500 dark:hover:text-red-400 hover:border-red-500/20 text-slate-500 dark:text-slate-400 text-xs font-bold rounded-xl flex items-center justify-center gap-1.5 transition-all"
                  >
                    <FiTrash2 size={13} /> Remove
                  </button>
                </motion.div>
              )
            })}
          </AnimatePresence>
        </div>

        {/* Detailed Course Matrix */}
        <h2 className="text-2xl font-extrabold mb-6 mt-12 flex items-center gap-2">
          <FiBookOpen className="text-primary-500" /> Course-wise Cutoffs Comparison
        </h2>

        <div className="glass overflow-hidden shadow-card border border-slate-200/50 dark:border-slate-800/50 rounded-3xl mb-16">
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-slate-50/50 dark:bg-slate-950/20 border-b border-slate-200/50 dark:border-slate-800/50">
                  <th className="p-5 text-sm font-extrabold text-slate-900 dark:text-slate-200">Course Specialization</th>
                  {loadedColleges.map((college) => (
                    <th key={college.code} className="p-5 text-sm font-extrabold text-slate-900 dark:text-slate-200">
                      <div className="truncate max-w-[200px]" title={college.college_name || college.name}>
                        {college.college_name || college.name}
                      </div>
                      <span className="text-[10px] text-slate-500 dark:text-slate-400 font-bold block mt-0.5">{college.code}</span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800/50">
                {uniqueCourses.map((course) => {
                  // Find the best (lowest) average cutoff for this course among these colleges
                  const courseCutoffs = loadedColleges.map((college) => {
                    const cMatch = college.courses?.find((c) => c.code === course.code)
                    return cMatch ? cMatch.avg_cutoff : null
                  }).filter(Boolean)
                  const bestCourseCutoff = courseCutoffs.length > 0 ? Math.min(...courseCutoffs) : null

                  return (
                    <tr key={course.code} className="hover:bg-slate-50/20 dark:hover:bg-slate-900/10 transition-colors">
                      <td className="p-5">
                        <p className="font-extrabold text-sm text-slate-900 dark:text-white">{course.name}</p>
                        <p className="text-[10px] text-slate-500 dark:text-slate-400 font-bold">{course.code}</p>
                      </td>
                      {loadedColleges.map((college) => {
                        const courseInfo = college.courses?.find((c) => c.code === course.code)
                        if (!courseInfo) {
                          return (
                            <td key={college.code} className="p-5 text-slate-400 dark:text-slate-600 text-xs">
                              <span className="inline-flex items-center gap-1 font-bold bg-slate-100 dark:bg-slate-900/40 text-slate-400 dark:text-slate-600 px-2 py-0.5 rounded">
                                <FiX size={12} /> Not Offered
                              </span>
                            </td>
                          )
                        }

                        const isBest = courseInfo.avg_cutoff === bestCourseCutoff

                        return (
                          <td key={college.code} className="p-5 text-slate-900 dark:text-slate-100 text-xs">
                            <div className="space-y-1">
                              <div className="flex items-center gap-1.5">
                                <span className={`font-extrabold text-sm ${isBest ? 'text-emerald-500' : ''}`}>
                                  {Math.round(courseInfo.avg_cutoff)?.toLocaleString()}
                                </span>
                                {isBest && (
                                  <span className="text-[9px] bg-emerald-500/10 text-emerald-500 font-extrabold px-1.5 py-0.5 rounded">
                                    Best
                                  </span>
                                )}
                              </div>
                              <p className="text-[10px] text-slate-400 font-medium">
                                Range: {courseInfo.min_cutoff?.toLocaleString()} - {courseInfo.max_cutoff?.toLocaleString()}
                              </p>
                            </div>
                          </td>
                        )
                      })}
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Comparison
