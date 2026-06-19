import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { useQuery } from '@tanstack/react-query'
import { SearchBar, Loader, useToast, ToastContainer } from '../components/common'
import { CourseCard } from '../components/cards'
import { courseService } from '../services/endpoints'
import { searchInArray } from '../utils/helpers'

const Courses = () => {
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedCourse, setSelectedCourse] = useState(null)
  const { toasts, removeToast, error: showError } = useToast()

  const { data: courses = [], isLoading } = useQuery({
    queryKey: ['courses'],
    queryFn: () => courseService.getAvailableCourses(),
    onError: (err) => showError(err.message),
  })

  const filteredCourses = searchInArray(courses, searchTerm, ['name', 'code'])

  if (isLoading) {
    return <Loader fullScreen />
  }

  return (
    <div className="min-h-screen bg-white dark:bg-gray-900 py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-12"
        >
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">Explore Courses</h1>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            Discover all available courses and their popularity
          </p>
        </motion.div>

        {/* Search */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mb-8 max-w-md"
        >
          <SearchBar
            onSearch={setSearchTerm}
            placeholder="Search courses..."
          />
        </motion.div>

        {/* Results */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mb-4"
        >
          <p className="text-gray-600 dark:text-gray-400">
            Found <span className="font-semibold">{filteredCourses.length}</span> courses
          </p>
        </motion.div>

        {/* Courses Grid */}
        {filteredCourses.length > 0 ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="grid md:grid-cols-2 lg:grid-cols-3 gap-6"
          >
            {filteredCourses.map((course, idx) => (
              <motion.div
                key={course.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: idx * 0.05 }}
              >
                <CourseCard
                  course={course}
                  onSelect={setSelectedCourse}
                  isSelected={selectedCourse?.id === course.id}
                />
              </motion.div>
            ))}
          </motion.div>
        ) : (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center py-12"
          >
            <p className="text-lg text-gray-600 dark:text-gray-400">
              No courses found
            </p>
          </motion.div>
        )}

        {/* Selected Course Details */}
        {selectedCourse && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-12 glass-card max-w-2xl mx-auto"
          >
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
              {selectedCourse.name}
            </h2>
            <div className="space-y-3 text-gray-700 dark:text-gray-300">
              <p><span className="font-semibold">Course Code:</span> {selectedCourse.code}</p>
              <p><span className="font-semibold">Colleges Offering:</span> {selectedCourse.college_count}</p>
              <p><span className="font-semibold">Popularity:</span> {selectedCourse.popularity}%</p>
              {selectedCourse.description && (
                <p><span className="font-semibold">Description:</span> {selectedCourse.description}</p>
              )}
            </div>
          </motion.div>
        )}
      </div>

      <ToastContainer toasts={toasts} removeToast={removeToast} />
    </div>
  )
}

export default Courses
