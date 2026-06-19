import React from 'react'
import { FiChevronLeft, FiChevronRight } from 'react-icons/fi'
import { Button } from './Button'

export const Pagination = ({ currentPage, totalPages, onPageChange }) => {
  const pages = []
  const maxPagesToShow = 5

  if (totalPages <= maxPagesToShow) {
    for (let i = 1; i <= totalPages; i++) {
      pages.push(i)
    }
  } else {
    const startPage = Math.max(1, currentPage - Math.floor(maxPagesToShow / 2))
    const endPage = Math.min(totalPages, startPage + maxPagesToShow - 1)

    if (endPage - startPage + 1 < maxPagesToShow) {
      const diff = maxPagesToShow - (endPage - startPage + 1)
      const adjustedStart = Math.max(1, startPage - diff)
      for (let i = adjustedStart; i <= endPage; i++) {
        pages.push(i)
      }
    } else {
      for (let i = startPage; i <= endPage; i++) {
        pages.push(i)
      }
    }
  }

  return (
    <div className="flex items-center justify-center gap-2 flex-wrap">
      <Button
        variant="outline"
        size="sm"
        onClick={() => onPageChange(currentPage - 1)}
        disabled={currentPage === 1}
      >
        <FiChevronLeft size={16} /> Prev
      </Button>

      {pages[0] > 1 && (
        <>
          <Button
            variant={1 === currentPage ? 'primary' : 'secondary'}
            size="sm"
            onClick={() => onPageChange(1)}
          >
            1
          </Button>
          {pages[0] > 2 && <span className="px-2 text-gray-500">...</span>}
        </>
      )}

      {pages.map((page) => (
        <Button
          key={page}
          variant={page === currentPage ? 'primary' : 'secondary'}
          size="sm"
          onClick={() => onPageChange(page)}
        >
          {page}
        </Button>
      ))}

      {pages[pages.length - 1] < totalPages && (
        <>
          {pages[pages.length - 1] < totalPages - 1 && <span className="px-2 text-gray-500">...</span>}
          <Button
            variant={totalPages === currentPage ? 'primary' : 'secondary'}
            size="sm"
            onClick={() => onPageChange(totalPages)}
          >
            {totalPages}
          </Button>
        </>
      )}

      <Button
        variant="outline"
        size="sm"
        onClick={() => onPageChange(currentPage + 1)}
        disabled={currentPage === totalPages}
      >
        Next <FiChevronRight size={16} />
      </Button>
    </div>
  )
}
