import React from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { FiArrowLeft, FiHome } from 'react-icons/fi'
import { Button } from '../components/common/Button'

const NotFound = () => {
  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white dark:from-gray-900 dark:to-gray-800 flex items-center justify-center px-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="text-center max-w-md"
      >
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.2, type: 'spring', stiffness: 200 }}
          className="text-9xl font-bold gradient-text mb-4"
        >
          404
        </motion.div>

        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-3">
          Page Not Found
        </h1>

        <p className="text-lg text-gray-600 dark:text-gray-400 mb-8">
          Oops! It seems you've wandered into the wrong college. The page you're looking for doesn't exist.
        </p>

        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Button variant="primary" size="lg" asChild>
            <Link to="/" className="flex items-center gap-2">
              <FiHome size={20} />
              Home
            </Link>
          </Button>
          <Button variant="outline" size="lg" asChild>
            <Link to="/predictor" className="flex items-center gap-2">
              <FiArrowLeft size={20} />
              Back
            </Link>
          </Button>
        </div>
      </motion.div>
    </div>
  )
}

export default NotFound
