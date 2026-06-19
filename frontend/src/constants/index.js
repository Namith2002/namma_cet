// Categories for KCET Rank prediction
export const CATEGORIES = [
  { value: 'GM', label: 'GM - General Merit' },
  { value: 'GMK', label: 'GMK - General Merit (Kannada)' },
  { value: 'GMR', label: 'GMR - General Merit (Religious)' },
  { value: 'SCG', label: 'SCG - SC General' },
  { value: 'SCK', label: 'SCK - SC Kannada' },
  { value: 'SCR', label: 'SCR - SC Religious' },
  { value: 'STG', label: 'STG - ST General' },
  { value: 'STK', label: 'STK - ST Kannada' },
  { value: 'STR', label: 'STR - ST Religious' },
  { value: '1G', label: '1G - Category 1 General' },
  { value: '1K', label: '1K - Category 1 Kannada' },
  { value: '1R', label: '1R - Category 1 Religious' },
  { value: '2AG', label: '2AG - Category 2A General' },
  { value: '2AK', label: '2AK - Category 2A Kannada' },
  { value: '2AR', label: '2AR - Category 2A Religious' },
  { value: '2BG', label: '2BG - Category 2B General' },
  { value: '2BK', label: '2BK - Category 2B Kannada' },
  { value: '2BR', label: '2BR - Category 2B Religious' },
  { value: '3AG', label: '3AG - Category 3A General' },
  { value: '3AK', label: '3AK - Category 3A Kannada' },
  { value: '3AR', label: '3AR - Category 3A Religious' },
  { value: '3BG', label: '3BG - Category 3B General' },
  { value: '3BK', label: '3BK - Category 3B Kannada' },
  { value: '3BR', label: '3BR - Category 3B Religious' },
]

export const REGIONS = [
  { value: 'General', label: 'General' },
  { value: 'HK', label: 'HK - Hyderabad Karnataka' },
]

export const STREAMS = [
  { value: 'Engineering', label: 'Engineering' },
  { value: 'Medical', label: 'Medical' },
  { value: 'Architecture', label: 'Architecture' },
]

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

export const MESSAGES = {
  LOADING: 'Loading...',
  ERROR: 'Something went wrong. Please try again.',
  SUCCESS: 'Operation successful!',
  NO_DATA: 'No data available',
  INVALID_INPUT: 'Please fill in all required fields',
}

export const SCORE_RANGES = {
  EXCELLENT: { min: 85, label: 'Excellent', color: 'green' },
  GOOD: { min: 70, max: 84, label: 'Good', color: 'blue' },
  AVERAGE: { min: 55, max: 69, label: 'Average', color: 'yellow' },
  POOR: { max: 54, label: 'Poor', color: 'red' },
}

export const PAGINATION = {
  DEFAULT_PAGE_SIZE: 10,
  OPTIONS: [10, 20, 50],
}
