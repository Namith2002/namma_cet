import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { SearchBar, FilterPanel, Loader, Pagination, useToast, ToastContainer } from '../components/common'
import { CollegeCard } from '../components/cards'
import { collegeService } from '../services/endpoints'
import { CATEGORIES, REGIONS } from '../constants'
import { searchInArray } from '../utils/helpers'
import { usePrediction } from '../context/PredictionContext'

const Colleges = () => {
  const [searchTerm, setSearchTerm] = useState('')
  const [currentPage, setCurrentPage] = useState(1)
  const [filters, setFilters] = useState({ category: '', region: '', course: '' })
  const { toasts, removeToast, error: showError } = useToast()
  const navigate = useNavigate()
  const { comparedColleges, clearComparedColleges } = usePrediction()

  const pageSize = 12

  const { data: colleges = [], isLoading } = useQuery({
    queryKey: ['colleges'],
    queryFn: () => collegeService.getAllColleges(),
    onError: (err) => showError(err.message),
  })

  // Apply search and filters
  let filteredColleges = searchInArray(colleges, searchTerm, ['name', 'code', 'course'])

  if (filters.category) {
    filteredColleges = filteredColleges.filter((c) => c.category === filters.category)
  }
  if (filters.region) {
    filteredColleges = filteredColleges.filter((c) => c.region === filters.region)
  }
  if (filters.course) {
    filteredColleges = filteredColleges.filter((c) => c.course === filters.course)
  }

  const totalPages = Math.ceil(filteredColleges.length / pageSize)
  const paginatedColleges = filteredColleges.slice(
    (currentPage - 1) * pageSize,
    currentPage * pageSize
  )

  const handleFilterChange = (key, value) => {
    setFilters((prev) => ({ ...prev, [key]: value }))
    setCurrentPage(1)
  }

  const handleReset = () => {
    setFilters({ category: '', region: '', course: '' })
    setCurrentPage(1)
  }

  if (isLoading) {
    return <Loader fullScreen />
  }

  return (
    <div className="min-h-screen py-12 relative overflow-hidden">
      {/* Background Decorative Gradients */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary-500/10 dark:bg-primary-500/5 rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-accent-blue/10 dark:bg-accent-blue/5 rounded-full blur-[120px] pointer-events-none" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-12 text-center md:text-left"
        >
          <h1 className="text-4xl md:text-5xl font-extrabold mb-4">
            Explore <span className="gradient-text">Colleges</span>
          </h1>
          <p className="text-lg text-slate-600 dark:text-slate-400 max-w-2xl">
            Browse through Karnataka's engineering and medical colleges. View cutoff trends, compare colleges, and find the perfect match.
          </p>
        </motion.div>

        {/* Search and Filters */}
        <div className="flex flex-col md:flex-row gap-4 mb-8 bg-slate-100/50 dark:bg-slate-900/30 p-4 rounded-3xl border border-slate-200/50 dark:border-slate-800/50 backdrop-blur-md">
          <div className="flex-1">
            <SearchBar
              onSearch={setSearchTerm}
              placeholder="Search colleges, codes, courses..."
            />
          </div>
          <FilterPanel
            filters={{
              category: { label: 'Category', options: CATEGORIES, value: filters.category },
              region: { label: 'Region', options: REGIONS, value: filters.region },
            }}
            onFilterChange={handleFilterChange}
            onReset={handleReset}
          />
        </div>

        {/* Results Info */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mb-6 flex justify-between items-center px-2"
        >
          <p className="text-sm text-slate-600 dark:text-slate-400">
            Showing <span className="font-bold text-slate-800 dark:text-slate-200">{paginatedColleges.length > 0 ? (currentPage - 1) * pageSize + 1 : 0}</span> to{' '}
            <span className="font-bold text-slate-800 dark:text-slate-200">{Math.min(currentPage * pageSize, filteredColleges.length)}</span> of{' '}
            <span className="font-extrabold text-primary-500">{filteredColleges.length}</span> colleges
          </p>
        </motion.div>

        {/* Colleges Grid */}
        {paginatedColleges.length > 0 ? (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12"
            >
              {paginatedColleges.map((college, idx) => (
                <motion.div
                  key={college.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: idx * 0.03 }}
                >
                  <CollegeCard
                    college={college}
                    onViewDetails={() => {}}
                  />
                </motion.div>
              ))}
            </motion.div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex justify-center mt-12">
                <Pagination
                  currentPage={currentPage}
                  totalPages={totalPages}
                  onPageChange={setCurrentPage}
                />
              </div>
            )}
          </>
        ) : (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center py-20 glass rounded-3xl"
          >
            <p className="text-lg text-slate-600 dark:text-slate-400 font-bold">
              No colleges found matching your criteria
            </p>
            <button
              onClick={handleReset}
              className="mt-4 px-6 py-2 bg-gradient-primary text-white font-bold rounded-xl shadow-glow-purple"
            >
              Reset Filters
            </button>
          </motion.div>
        )}
      </div>

      {/* Floating Comparison Drawer/Bar */}
      <AnimatePresence>
        {comparedColleges.length > 0 && (
          <motion.div
            initial={{ y: 100, opacity: 0, x: '-50%' }}
            animate={{ y: 0, opacity: 1, x: '-50%' }}
            exit={{ y: 100, opacity: 0, x: '-50%' }}
            className="fixed bottom-6 left-1/2 z-50 w-full max-w-xl px-4"
          >
            <div className="bg-slate-900/90 dark:bg-slate-950/95 backdrop-blur-xl border border-slate-800/80 rounded-3xl p-4 shadow-glow-purple flex items-center justify-between gap-4 text-white">
              <div className="flex items-center gap-3">
                <div className="relative flex h-3 w-3">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-3 w-3 bg-primary-500"></span>
                </div>
                <div>
                  <p className="font-extrabold text-sm text-slate-100">
                    {comparedColleges.length} / 3 Colleges Selected
                  </p>
                  <p className="text-[10px] text-slate-400">Compare their details side-by-side</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={clearComparedColleges}
                  className="px-3 py-2 text-xs font-bold text-slate-400 hover:text-white transition-colors hover:bg-slate-800/50 rounded-xl"
                >
                  Clear All
                </button>
                <button
                  onClick={() => navigate('/comparison')}
                  className="px-5 py-2.5 bg-gradient-primary rounded-xl text-xs font-extrabold shadow-glow-purple hover:scale-[1.02] active:scale-[0.98] transition-all"
                >
                  Compare Now
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <ToastContainer toasts={toasts} removeToast={removeToast} />
    </div>
  )
}

export default Colleges
