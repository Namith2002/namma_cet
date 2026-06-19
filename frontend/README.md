# NammaCET Frontend

A modern, production-ready React application for KCET Rank Prediction and College Allocation.

## 🚀 Features

- **Accurate Rank Predictions** - ML-powered predictions using historical KCET data
- **College Explorer** - Browse and filter colleges with detailed information
- **Course Analytics** - Explore popular courses and statistics
- **Interactive Dashboard** - Real-time analytics and visualizations
- **Dark Mode** - Beautiful dark mode support
- **Responsive Design** - Mobile, tablet, and desktop optimized
- **Performance Optimized** - Code splitting, lazy loading, and caching

## 📋 Tech Stack

- **Frontend Framework**: React 18 with Vite
- **Styling**: Tailwind CSS with Dark Mode
- **UI Animations**: Framer Motion
- **State Management**: Context API + TanStack Query
- **Forms**: React Hook Form
- **Charts**: Recharts
- **HTTP Client**: Axios with interceptors
- **Routing**: React Router DOM v6
- **Icons**: React Icons

## 🛠️ Installation

### Prerequisites
- Node.js 16+ 
- npm or yarn

### Setup

1. **Clone and Install**
```bash
cd frontend
npm install
```

2. **Environment Configuration**
```bash
cp .env.example .env.local
```

Edit `.env.local`:
```env
VITE_API_BASE_URL=http://localhost:8000
VITE_APP_NAME=NammaCET
```

3. **Start Development Server**
```bash
npm run dev
```

The app will open at `http://localhost:5173`

## 📦 Build

```bash
npm run build
```

Optimized production build will be in the `dist/` folder.

## 🏗️ Project Structure

```
src/
├── assets/              # Static assets
├── components/
│   ├── common/         # Reusable UI components
│   ├── forms/          # Form components
│   ├── charts/         # Chart components
│   ├── cards/          # Card components
│   ├── tables/         # Table components
│   └── layout/         # Layout components
├── pages/              # Page components
├── services/           # API services
├── hooks/              # Custom React hooks
├── context/            # Context API setup
├── utils/              # Helper functions
├── constants/          # Constants
├── layouts/            # Page layouts
└── App.jsx             # Main app component
```

## 🎨 Components

### Common Components
- **Button** - Versatile button with variants
- **Input** - Enhanced input field
- **Select** - Dropdown selector
- **Modal** - Modal dialog
- **Loader** - Loading spinner
- **Toast** - Toast notifications
- **Pagination** - Page navigation
- **SearchBar** - Search with debounce
- **FilterPanel** - Advanced filters

### Card Components
- **StatCard** - Statistics display
- **CollegeCard** - College information
- **CourseCard** - Course details

### Chart Components
- **SimpleBarChart** - Bar chart
- **SimpleLineChart** - Line chart
- **SimplePieChart** - Pie chart
- **AreaChartComponent** - Area chart
- **RadarChartComponent** - Radar chart
- **MultiSeriesBarChart** - Multi-series chart

## 📄 Pages

- **Landing** - Home page with features and CTA
- **Predictor** - KCET rank prediction
- **Colleges** - Browse and filter colleges
- **Courses** - Explore available courses
- **Analytics** - Data visualizations and insights
- **Allocation** - Find eligible colleges by rank
- **Comparison** - Compare colleges side-by-side
- **About** - About the platform

## 🔌 API Integration

API calls are centralized in `services/endpoints.js`:

```javascript
import { predictionService, collegeService } from './services/endpoints'

// Predict rank
const result = await predictionService.predictRank(data)

// Get colleges
const colleges = await collegeService.getAllColleges()
```

### Available Endpoints

- `POST /predict` - Predict KCET rank
- `POST /allocate` - Allocate colleges
- `GET /colleges` - Get all colleges
- `GET /courses` - Get available courses
- `GET /analytics` - Get analytics data

## 🎯 Context API

### ThemeContext
- `isDark` - Current theme
- `toggleTheme()` - Switch theme

### PredictionContext
- `predictionResult` - Stored prediction
- `savePrediction()` - Save prediction
- `allocationResults` - Stored allocation

## 🎣 Custom Hooks

- `useLocalStorage()` - Persistent state
- `useDebounce()` - Debounced values
- `useAsync()` - Async operations
- `useIntersectionObserver()` - Intersection detection
- `useWindowSize()` - Window dimensions

## 🌙 Dark Mode

Theme automatically detects system preference and allows manual toggle.

## 📱 Responsive Breakpoints

- `sm` - 640px
- `md` - 768px
- `lg` - 1024px
- `xl` - 1280px
- `2xl` - 1536px

## 🚀 Deployment

### Vercel
```bash
npm run build
vercel
```

### Docker
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 5173
CMD ["npm", "run", "preview"]
```

## 📝 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_BASE_URL` | Backend API URL | `http://localhost:8000` |
| `VITE_APP_NAME` | App name | `NammaCET` |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

MIT License

## 📞 Support

For issues and questions, please reach out to info@nammacet.com

---

**Built with ❤️ for Karnataka students**
