import React from 'react'
import { motion } from 'framer-motion'

export const Loader = ({ size = 'md', fullScreen = false }) => {
  const sizeClasses = {
    sm: 'w-6 h-6',
    md: 'w-10 h-10',
    lg: 'w-16 h-16',
  }

  const loader = (
    <motion.div
      className={`${sizeClasses[size]} border-4 border-blue-200 dark:border-blue-900 border-t-blue-600 rounded-full`}
      animate={{ rotate: 360 }}
      transition={{ duration: 1, repeat: Infinity, easing: 'linear' }}
    />
  )

  if (fullScreen) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-white dark:bg-gray-900">
        {loader}
      </div>
    )
  }

  return <div className="flex items-center justify-center p-4">{loader}</div>
}

export const SkeletonLoader = ({ count = 3, type = 'card' }) => {
  if (type === 'card') {
    return (
      <div className="space-y-4">
        {Array.from({ length: count }).map((_, i) => (
          <motion.div
            key={i}
            className="h-24 bg-gray-200 dark:bg-gray-700 rounded-lg"
            animate={{ opacity: [0.5, 1, 0.5] }}
            transition={{ duration: 1.5, repeat: Infinity }}
          />
        ))}
      </div>
    )
  }

  return (
    <div className="space-y-2">
      {Array.from({ length: count }).map((_, i) => (
        <motion.div
          key={i}
          className="h-4 bg-gray-200 dark:bg-gray-700 rounded"
          animate={{ opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 1.5, repeat: Infinity }}
        />
      ))}
    </div>
  )
}
