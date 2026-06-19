import React from 'react'
import { FiSearch, FiX } from 'react-icons/fi'
import { useDebounce } from '../../hooks'

export const SearchBar = ({ onSearch, placeholder = 'Search...', className = '' }) => {
  const [query, setQuery] = React.useState('')
  const debouncedQuery = useDebounce(query, 300)

  React.useEffect(() => {
    onSearch(debouncedQuery)
  }, [debouncedQuery, onSearch])

  const handleClear = () => {
    setQuery('')
  }

  return (
    <div className={`relative w-full ${className}`}>
      <FiSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder={placeholder}
        className="w-full pl-10 pr-10 py-2.5 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600
          rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent
          dark:text-white placeholder-gray-400 dark:placeholder-gray-500"
      />
      {query && (
        <button
          onClick={handleClear}
          className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
        >
          <FiX size={18} />
        </button>
      )}
    </div>
  )
}
