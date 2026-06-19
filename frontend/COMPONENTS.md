## 📚 Components Documentation

### Common UI Components

#### Button
Versatile button component with multiple variants and sizes.

```jsx
import { Button } from '@/components/common'

<Button variant="primary" size="md">
  Click Me
</Button>

// Props
variant: 'primary' | 'secondary' | 'outline' | 'danger' | 'success'
size: 'sm' | 'md' | 'lg' | 'xl'
disabled: boolean
isLoading: boolean
```

#### Input
Enhanced text input with validation.

```jsx
<Input
  label="Enter Name"
  type="text"
  error={error?.message}
  required
/>
```

#### Select
Dropdown selector component.

```jsx
<Select
  label="Choose Category"
  options={[
    { value: 'GM', label: 'General Merit' },
    { value: 'SC', label: 'Scheduled Caste' }
  ]}
  error={error?.message}
/>
```

#### Modal
Modal dialog for overlays and confirmations.

```jsx
<Modal
  isOpen={isOpen}
  onClose={handleClose}
  title="Confirm Action"
  size="md"
>
  <p>Are you sure?</p>
</Modal>
```

#### Loader
Loading spinner with optional full screen mode.

```jsx
<Loader size="md" />
<Loader fullScreen={true} />
```

#### Toast Notifications
Toast messages for user feedback.

```jsx
const { toasts, removeToast, success, error } = useToast()

success('Operation completed!')
error('Something went wrong')

<ToastContainer toasts={toasts} removeToast={removeToast} />
```

#### Pagination
Pagination component for large datasets.

```jsx
<Pagination
  currentPage={page}
  totalPages={totalPages}
  onPageChange={setPage}
/>
```

#### SearchBar
Debounced search input.

```jsx
<SearchBar
  onSearch={(term) => handleSearch(term)}
  placeholder="Search colleges..."
/>
```

#### FilterPanel
Advanced filter dropdown.

```jsx
<FilterPanel
  filters={{
    category: { label: 'Category', options: [...], value: '' },
    region: { label: 'Region', options: [...], value: '' }
  }}
  onFilterChange={(key, value) => {}}
  onReset={() => {}}
/>
```

---

### Card Components

#### StatCard
Statistics display card with icon and optional trend.

```jsx
import { StatCard } from '@/components/cards'

<StatCard
  title="Total Colleges"
  value="260+"
  icon={FiUsers}
  trend={{ positive: true, value: 12 }}
  color="blue"
/>

// Colors: blue, green, purple, orange
```

#### CollegeCard
College information card with eligibility status.

```jsx
import { CollegeCard } from '@/components/cards'

<CollegeCard
  college={collegeData}
  onViewDetails={(id) => {}}
  onCompare={(id) => {}}
  userRank={5000}
  category="GM"
/>
```

#### CourseCard
Selectable course card with popularity bar.

```jsx
import { CourseCard } from '@/components/cards'

<CourseCard
  course={courseData}
  onSelect={(id) => {}}
  isSelected={true}
/>
```

---

### Chart Components

#### SimpleBarChart
Basic bar chart using Recharts.

```jsx
import { SimpleBarChart } from '@/components/charts'

<SimpleBarChart
  data={[
    { name: 'Engineering', value: 150 },
    { name: 'Medical', value: 100 }
  ]}
  title="College Distribution"
  dataKey="value"
  xAxisKey="name"
/>
```

#### SimpleLineChart
Line chart for trends.

```jsx
<SimpleLineChart
  data={data}
  title="Admission Trend"
  dataKey="admissions"
  xAxisKey="year"
/>
```

#### SimplePieChart
Pie chart for distributions.

```jsx
<SimplePieChart
  data={data}
  title="Category Distribution"
  dataKey="count"
/>
```

#### AreaChartComponent
Area chart for cumulative data.

```jsx
<AreaChartComponent
  data={data}
  title="Enrollment Trend"
  dataKey="enrollments"
  xAxisKey="month"
/>
```

#### RadarChartComponent
Radar chart for multi-dimensional comparison.

```jsx
<RadarChartComponent
  data={data}
  title="College Comparison"
  dataKey="score"
/>
```

#### MultiSeriesBarChart
Bar chart with multiple data series.

```jsx
<MultiSeriesBarChart
  data={data}
  title="Category-wise Analysis"
  series={[{ key: 'general' }, { key: 'reserved' }]}
  xAxisKey="category"
/>
```

---

### Form Components

#### RankPredictorForm
Complex form for KCET rank prediction.

```jsx
import { RankPredictorForm } from '@/components/forms'

<RankPredictorForm
  onSubmit={(data) => handlePredict(data)}
  isLoading={isPending}
/>

// Returns object with:
// {
//   physics_kcet, chemistry_kcet, mathematics_kcet, biology_kcet,
//   physics_theory, chemistry_theory, mathematics_theory, biology_theory,
//   category, region
// }
```

---

### Table Components

#### DataTable
Responsive data table with sorting and loading states.

```jsx
import { DataTable } from '@/components/tables'

const columns = [
  { key: 'name', label: 'Name', width: '200px' },
  { key: 'rank', label: 'Rank', width: '100px', render: (val) => `#${val}` },
  { key: 'course', label: 'Course', width: '200px' }
]

<DataTable
  columns={columns}
  data={collegeList}
  loading={isLoading}
/>
```

---

### Layout Components

#### Navbar
Navigation header with theme toggle.

```jsx
// Auto-integrated in main App
// Features: Logo, Nav links, Theme toggle, Mobile menu
```

#### Footer
Footer with links and social media.

```jsx
// Auto-integrated in main App
// Features: Links, Social icons, Copyright
```

---

## 🎯 Custom Hooks

### useLocalStorage
Persistent state using localStorage.

```jsx
const [user, setUser] = useLocalStorage('user', null)
```

### useDebounce
Debounced value for search/filter.

```jsx
const debouncedSearch = useDebounce(searchTerm, 300)
```

### useAsync
Async operation management.

```jsx
const { execute, status, value, error } = useAsync(fetchData, false)
await execute()
```

### useIntersectionObserver
Detect when element enters viewport.

```jsx
const ref = useRef()
const isVisible = useIntersectionObserver(ref)
```

### useWindowSize
Get window dimensions with resize tracking.

```jsx
const { width, height } = useWindowSize()
```

### useOnClickOutside
Detect clicks outside element.

```jsx
const ref = useRef()
useOnClickOutside(ref, () => closeMenu())
```

---

## 🎨 Theme Context

Access theme state throughout app.

```jsx
import { useTheme } from '@/context/ThemeContext'

const { isDark, toggleTheme } = useTheme()
```

---

## 🔮 Prediction Context

Manage prediction results globally.

```jsx
import { usePrediction } from '@/context/PredictionContext'

const {
  predictionResult,
  savePrediction,
  clearPrediction,
  allocationResults,
  saveAllocation
} = usePrediction()
```

---

## 📋 Constants

### Categories
```javascript
import { CATEGORIES } from '@/constants'
// Returns array of category options
```

### Regions
```javascript
import { REGIONS } from '@/constants'
// GM, HK
```

### API Base URL
```javascript
import { API_BASE_URL } from '@/constants'
// http://localhost:8000
```

---

## 🎪 Example Page Implementation

```jsx
import { useQuery } from '@tanstack/react-query'
import { useToast, ToastContainer } from '@/components/common'
import { StatCard, CollegeCard } from '@/components/cards'
import { collegeService } from '@/services/endpoints'

const MyPage = () => {
  const { toasts, removeToast, success, error } = useToast()

  const { data, isLoading } = useQuery({
    queryKey: ['colleges'],
    queryFn: () => collegeService.getAllColleges(),
    onError: (err) => error(err.message)
  })

  return (
    <>
      <div className="grid gap-6">
        {data?.map(college => (
          <CollegeCard
            key={college.id}
            college={college}
            onViewDetails={() => success('College selected!')}
            onCompare={() => {}}
          />
        ))}
      </div>
      <ToastContainer toasts={toasts} removeToast={removeToast} />
    </>
  )
}
```

---

## ✨ Styling Patterns

### Glass Effect
```jsx
<div className="glass-card">
  // Glassmorphic card
</div>
```

### Gradient Text
```jsx
<h1 className="gradient-text">
  Special Title
</h1>
```

### Responsive Grid
```jsx
<div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
  {/* Items */}
</div>
```

### Animation
```jsx
import { motion } from 'framer-motion'

<motion.div
  initial={{ opacity: 0, y: 20 }}
  animate={{ opacity: 1, y: 0 }}
>
  Animated content
</motion.div>
```

---

**All components are production-ready and fully typed with JSDoc comments.**
