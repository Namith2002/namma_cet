// Utility functions for calculations and formatting

export const calculatePercentage = (value, max = 100) => {
  return ((value / max) * 100).toFixed(2);
};

export const getScoreColor = (score, max = 100) => {
  const percentage = (score / max) * 100;
  if (percentage >= 85) return 'text-green-600';
  if (percentage >= 70) return 'text-blue-600';
  if (percentage >= 55) return 'text-yellow-600';
  return 'text-red-600';
};

export const getScoreBgColor = (score, max = 100) => {
  const percentage = (score / max) * 100;
  if (percentage >= 85) return 'bg-green-100 dark:bg-green-900/20';
  if (percentage >= 70) return 'bg-blue-100 dark:bg-blue-900/20';
  if (percentage >= 55) return 'bg-yellow-100 dark:bg-yellow-900/20';
  return 'bg-red-100 dark:bg-red-900/20';
};

export const formatRank = (rank) => {
  if (!rank) return '--';
  return new Intl.NumberFormat('en-IN').format(Math.round(rank));
};

export const formatNumber = (num) => {
  return new Intl.NumberFormat('en-IN').format(num);
};

export const formatScore = (score) => {
  return Number(score).toFixed(2);
};

export const capitalizeFirst = (str) => {
  if (!str) return '';
  return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
};

export const slugify = (str) => {
  return str.toLowerCase().replace(/\s+/g, '-').replace(/[^\w-]+/g, '');
};

export const debounce = (func, wait) => {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
};

export const throttle = (func, limit) => {
  let inThrottle;
  return function(...args) {
    if (!inThrottle) {
      func.apply(this, args);
      inThrottle = true;
      setTimeout(() => (inThrottle = false), limit);
    }
  };
};

export const clamp = (num, min, max) => {
  return Math.max(min, Math.min(num, max));
};

export const getInitials = (name) => {
  return name
    .split(' ')
    .map((word) => word[0])
    .join('')
    .toUpperCase()
    .slice(0, 2);
};

export const calculateEligibility = (userRank, cutoffRank) => {
  if (!userRank || !cutoffRank) return null;
  return userRank <= cutoffRank;
};

export const getEligibilityStatus = (eligible) => {
  if (eligible === null) return 'N/A';
  return eligible ? 'Eligible' : 'Not Eligible';
};

export const getEligibilityColor = (eligible) => {
  if (eligible === null) return 'gray';
  return eligible ? 'green' : 'red';
};

export const sortBy = (arr, key, ascending = true) => {
  const sorted = [...arr].sort((a, b) => {
    if (ascending) {
      return a[key] > b[key] ? 1 : -1;
    }
    return a[key] < b[key] ? 1 : -1;
  });
  return sorted;
};

export const filterBy = (arr, key, value) => {
  return arr.filter((item) => item[key] === value);
};

export const searchInArray = (arr, searchTerm, searchKeys = []) => {
  if (!searchTerm) return arr;
  const lowerSearch = searchTerm.toLowerCase();
  return arr.filter((item) => {
    return searchKeys.some((key) => {
      const value = item[key];
      if (value === null || value === undefined) return false;
      return value.toString().toLowerCase().includes(lowerSearch);
    });
  });
};

export const getPageRange = (currentPage, totalPages, pageSize = 5) => {
  const halfSize = Math.floor(pageSize / 2);
  let start = Math.max(1, currentPage - halfSize);
  let end = Math.min(totalPages, start + pageSize - 1);

  if (end - start + 1 < pageSize) {
    start = Math.max(1, end - pageSize + 1);
  }

  return Array.from({ length: end - start + 1 }, (_, i) => start + i);
};
