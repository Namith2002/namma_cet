import apiClient from './api'

const mapCollege = (c) => {
  if (!c) return c;
  return {
    id: c.college_code || c.code || '',
    name: c.college_name || c.name || '',
    code: c.college_code || c.code || '',
    course: c.course_name || c.course_code || c.course || '',
    cutoff_rank: c.cutoff_rank !== undefined ? c.cutoff_rank : (c.min_rank || 0),
    ...c
  }
}

const mapCourse = (c) => {
  if (!c) return c;
  return {
    id: c.course_code || c.code || '',
    name: c.course_name || c.name || '',
    code: c.course_code || c.code || '',
    popularity: c.popularity || 0,
    college_count: c.college_count || 0,
    ...c
  }
}

export const predictionService = {
  predictRank: (data) => {
    const toInt = (val) => {
      if (val === null || val === undefined || isNaN(val)) return 0
      return Math.round(Number(val)) || 0
    }
    const payload = {
      physics_kcet: toInt(data.kcet_physics),
      chemistry_kcet: toInt(data.kcet_chemistry),
      mathematics_kcet: toInt(data.kcet_mathematics),
      biology_kcet: toInt(data.kcet_biology),
      physics_theory: toInt(data.theory_physics),
      chemistry_theory: toInt(data.theory_chemistry),
      mathematics_theory: toInt(data.theory_mathematics),
      biology_theory: toInt(data.theory_biology),
      category_code: data.category || 'GM',
      category_type: data.region || 'General',
    }
    return apiClient.post('/predict', payload).then((res) => {
      if (res && res.combined_total !== undefined) {
        res.combined_score = res.combined_total
      }
      if (res && Array.isArray(res.eligible_colleges)) {
        res.eligible_colleges = res.eligible_colleges.map(mapCollege)
      }
      return res
    })
  },
  getHistoricalData: () => apiClient.get('/historical-data'),
}

export const allocationService = {
  allocateCollege: (data) => {
    const payload = {
      rank: Math.round(Number(data.rank)) || 0,
      category_code: data.category || 'GM',
      category_type: data.region || 'General',
    }
    return apiClient.post('/allocate', payload).then((res) => {
      const colleges = res.eligible_colleges || []
      return colleges.map(mapCollege)
    })
  },
  getCollegesByRank: (rank, category, region) =>
    apiClient.get('/colleges', {
      params: { rank, category, region },
    }),
}

export const collegeService = {
  getAllColleges: () => apiClient.get('/colleges').then((res) => {
    const colleges = res.colleges || []
    return colleges.map(mapCollege)
  }),
  getCollegeDetails: (collegeId) => apiClient.get(`/colleges/${collegeId}`),
  searchColleges: (query, filters) =>
    apiClient.get('/colleges/search', { params: { q: query, ...filters } }),
  compareColleges: (collegeIds) =>
    apiClient.post('/colleges/compare', { college_ids: collegeIds }),
}

export const courseService = {
  getAvailableCourses: () => apiClient.get('/courses').then((res) => {
    const courses = res.courses || []
    // Add popularity if missing to sort
    const maxCount = Math.max(...courses.map(c => c.college_count || 1), 1)
    return courses.map(c => {
      const popularity = Math.round(((c.college_count || 0) / maxCount) * 100)
      return mapCourse({ popularity, ...c })
    })
  }),
  getCourseDetails: (courseId) => apiClient.get(`/courses/${courseId}`),
  getCourseByCutoff: (cutoff) => apiClient.get('/courses/by-cutoff', { params: { cutoff } }),
}

export const analyticsService = {
  getAnalytics: () => apiClient.get('/analytics'),
  getCoursPopularity: () => apiClient.get('/analytics/course-popularity'),
  getCutoffDistribution: () => apiClient.get('/analytics/cutoff-distribution'),
  getCategoryAnalysis: () => apiClient.get('/analytics/category-analysis'),
  getRankDistribution: () => apiClient.get('/analytics/rank-distribution'),
  getAdmissionTrends: () => apiClient.get('/analytics/admission-trends'),
}
