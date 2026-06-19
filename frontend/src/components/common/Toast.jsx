import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  FiCheckCircle,
  FiAlertCircle,
  FiX,
  FiInfo
} from 'react-icons/fi'

const toastVariants = {
  initial: { opacity: 0, y: 50, scale: 0.9 },
  animate: { opacity: 1, y: 0, scale: 1 },
  exit: { opacity: 0, y: 50, scale: 0.9 }
}

// Export Toast component
export const Toast = ({
  message,
  type = 'info',
  onClose,
  duration = 3000
}) => {
  const typeClasses = {
    success:
      'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800',
    error:
      'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800',
    info:
      'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800',
    warning:
      'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800'
  }

  const iconClasses = {
    success: 'text-green-600 dark:text-green-400',
    error: 'text-red-600 dark:text-red-400',
    info: 'text-blue-600 dark:text-blue-400',
    warning: 'text-yellow-600 dark:text-yellow-400'
  }

  const icons = {
    success: <FiCheckCircle size={20} />,
    error: <FiAlertCircle size={20} />,
    info: <FiInfo size={20} />,
    warning: <FiAlertCircle size={20} />
  }

  useEffect(() => {
    const timer = setTimeout(() => {
      onClose?.()
    }, duration)

    return () => clearTimeout(timer)
  }, [onClose, duration])

  return (
    <motion.div
      variants={toastVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      className={`flex items-center gap-3 px-4 py-3 rounded-lg border shadow-lg backdrop-blur-sm ${typeClasses[type]}`}
    >
      <span className={iconClasses[type]}>
        {icons[type]}
      </span>

      <p className="flex-1 text-sm font-medium text-gray-900 dark:text-white">
        {message}
      </p>

      <button
        onClick={onClose}
        className="text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 transition-colors"
      >
        <FiX size={16} />
      </button>
    </motion.div>
  )
}

// Export ToastContainer
export const ToastContainer = ({
  toasts = [],
  removeToast
}) => {
  return (
    <div className="fixed bottom-4 right-4 z-[9999] space-y-2">
      <AnimatePresence>
        {toasts.map((toast) => (
          <Toast
            key={toast.id}
            message={toast.message}
            type={toast.type}
            duration={toast.duration}
            onClose={() => removeToast(toast.id)}
          />
        ))}
      </AnimatePresence>
    </div>
  )
}

// Export useToast hook
export const useToast = () => {
  const [toasts, setToasts] = useState([])

  const addToast = (
    message,
    type = 'info',
    duration = 3000
  ) => {
    const id = Date.now() + Math.random()

    setToasts((prev) => [
      ...prev,
      {
        id,
        message,
        type,
        duration
      }
    ])

    return id
  }

  const removeToast = (id) => {
    setToasts((prev) =>
      prev.filter((toast) => toast.id !== id)
    )
  }

  return {
    toasts,
    addToast,
    removeToast,
    success: (message, duration) =>
      addToast(message, 'success', duration),

    error: (message, duration) =>
      addToast(message, 'error', duration),

    info: (message, duration) =>
      addToast(message, 'info', duration),

    warning: (message, duration) =>
      addToast(message, 'warning', duration)
  }
}