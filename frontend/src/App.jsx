import React from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { QueryClientProvider, QueryClient } from '@tanstack/react-query'
import { Navbar, Footer } from './components/layout'
import { ThemeProvider } from './context/ThemeContext'
import { PredictionProvider } from './context/PredictionContext'

// Pages
import Landing from './pages/Landing'
import Predictor from './pages/Predictor'
import Colleges from './pages/Colleges'
import Courses from './pages/Courses'
import Analytics from './pages/Analytics'
import About from './pages/About'
import Allocation from './pages/Allocation'
import Comparison from './pages/Comparison'
import NotFound from './pages/NotFound'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      gcTime: 1000 * 60 * 10, // 10 minutes
    },
  },
})

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <PredictionProvider>
          <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
            <div className="flex flex-col min-h-screen bg-white dark:bg-gray-900">
              <Navbar />
              <main className="flex-grow">
                <Routes>
                  <Route path="/" element={<Landing />} />
                  <Route path="/predictor" element={<Predictor />} />
                  <Route path="/colleges" element={<Colleges />} />
                  <Route path="/courses" element={<Courses />} />
                  <Route path="/analytics" element={<Analytics />} />
                  <Route path="/about" element={<About />} />
                  <Route path="/allocation" element={<Allocation />} />
                  <Route path="/comparison" element={<Comparison />} />
                  <Route path="*" element={<NotFound />} />
                </Routes>
              </main>
              <Footer />
            </div>
          </BrowserRouter>
        </PredictionProvider>
      </ThemeProvider>
    </QueryClientProvider>
  )
}

export default App
