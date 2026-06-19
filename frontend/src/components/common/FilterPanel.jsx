import React from 'react'
import { FiFilter, FiX } from 'react-icons/fi'
import { Button } from './Button'
import { Select } from './Select'

export const FilterPanel = ({ filters, onFilterChange, onReset, className = '' }) => {
  const [isOpen, setIsOpen] = React.useState(false)

  return (
    <div className={`relative ${className}`}>
      <Button
        variant="outline"
        size="md"
        onClick={() => setIsOpen(!isOpen)}
        className="gap-2"
      >
        <FiFilter size={18} />
        Filters
      </Button>

      {isOpen && (
        <div className="absolute top-full mt-2 right-0 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600
          rounded-lg shadow-lg p-4 w-80 z-40">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-gray-900 dark:text-white">Filters</h3>
            <button
              onClick={() => setIsOpen(false)}
              className="text-gray-500 hover:text-gray-700"
            >
              <FiX size={20} />
            </button>
          </div>

          <div className="space-y-4">
            {Object.entries(filters).map(([key, config]) => (
              <Select
                key={key}
                label={config.label}
                options={config.options}
                value={config.value}
                onChange={(e) => onFilterChange(key, e.target.value)}
              />
            ))}
          </div>

          <div className="flex gap-2 mt-6">
            <Button
              variant="outline"
              size="sm"
              className="flex-1"
              onClick={() => {
                onReset()
                setIsOpen(false)
              }}
            >
              Reset
            </Button>
            <Button
              variant="primary"
              size="sm"
              className="flex-1"
              onClick={() => setIsOpen(false)}
            >
              Apply
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}
