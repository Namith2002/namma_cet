import React from 'react'
import { motion } from 'framer-motion'

export const Button = React.forwardRef(
  (
    {
      children,
      className = '',
      variant = 'primary',
      size = 'md',
      disabled = false,
      isLoading = false,
      asChild = false,
      ...props
    },
    ref
  ) => {
    const baseClasses =
      'font-semibold rounded-lg transition-all duration-200 flex items-center justify-center gap-2 whitespace-nowrap'

    const sizeClasses = {
      sm: 'px-3 py-1.5 text-sm',
      md: 'px-4 py-2.5 text-base',
      lg: 'px-6 py-3 text-lg',
      xl: 'px-8 py-4 text-xl',
    }

    const variantClasses = {
      primary: 'bg-blue-600 hover:bg-blue-700 text-white shadow-md hover:shadow-lg disabled:bg-blue-400',
      secondary: 'bg-gray-200 hover:bg-gray-300 text-gray-900 dark:bg-gray-700 dark:hover:bg-gray-600 dark:text-white',
      outline: 'border-2 border-blue-600 text-blue-600 hover:bg-blue-50 dark:hover:bg-blue-900/20',
      danger: 'bg-red-600 hover:bg-red-700 text-white',
      success: 'bg-green-600 hover:bg-green-700 text-white',
    }

    const buttonClasses = `${baseClasses} ${sizeClasses[size]} ${variantClasses[variant]} ${className}`

    if (asChild && React.isValidElement(children)) {
      return React.cloneElement(children, {
        className: `${buttonClasses} ${children.props.className || ''}`,
        disabled: disabled || isLoading,
        ref,
      })
    }

    return (
      <motion.button
        ref={ref}
        className={buttonClasses}
        disabled={disabled || isLoading}
        whileHover={{ scale: disabled ? 1 : 1.02 }}
        whileTap={{ scale: disabled ? 1 : 0.98 }}
        {...props}
      >
        {isLoading && (
          <motion.div
            className="w-4 h-4 border-2 border-current border-t-transparent rounded-full"
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, easing: 'linear' }}
          />
        )}
        {children}
      </motion.button>
    )
  }
)

Button.displayName = 'Button'
