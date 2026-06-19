import React from 'react'
import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import { FiArrowRight, FiBook, FiBarChart3, FiUsers, FiTarget, FiZap, FiTrendingUp } from 'react-icons/fi'
import { Button } from '../components/common/Button'
import { StatCard } from '../components/cards'

const Landing = () => {
  const features = [
    {
      icon: FiTarget,
      title: 'Accurate Predictions',
      description: 'ML-powered rank prediction using historical data and advanced algorithms',
    },
    {
      icon: FiBook,
      title: 'College Explorer',
      description: 'Browse all colleges, courses, and explore detailed information',
    },
    {
      icon: FiBarChart3,
      title: 'Analytics Dashboard',
      description: 'Analyze trends, cutoffs, and admission patterns with interactive charts',
    },
    {
      icon: FiZap,
      title: 'Quick Results',
      description: 'Get instant results and recommendations personalized for you',
    },
  ]

  const stats = [
    { title: 'Total Colleges', value: '260+', icon: FiUsers, color: 'blue' },
    { title: 'Total Courses', value: '500+', icon: FiBook, color: 'purple' },
    { title: 'Success Rate', value: '95%', icon: FiTrendingUp, color: 'green' },
    { title: 'Students Helped', value: '10K+', icon: FiUsers, color: 'orange' },
  ]

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white dark:from-gray-900 dark:to-gray-800">
      {/* Hero Section */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 md:py-32">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center max-w-3xl mx-auto"
        >
          <h1 className="text-5xl md:text-6xl font-bold text-gray-900 dark:text-white mb-6">
            Predict Your{' '}
            <span className="gradient-text">KCET Rank</span>
            <br />& Find Your Dream College
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-400 mb-8">
            Use our advanced ML models to predict your rank, explore eligible colleges, and make informed decisions about your future education.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button variant="primary" size="lg">
              <Link to="/predictor" className="flex items-center gap-2">
                Predict Your Rank
                <FiArrowRight size={20} />
              </Link>
            </Button>
            <Button variant="outline" size="lg">
              <Link to="/colleges">Explore Colleges</Link>
            </Button>
          </div>
        </motion.div>
      </section>

      {/* Stats Section */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="grid md:grid-cols-4 gap-6">
          {stats.map((stat, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.1 }}
            >
              <StatCard {...stat} />
            </motion.div>
          ))}
        </div>
      </section>

      {/* Features Section */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">Why Choose NammaCET?</h2>
          <p className="text-xl text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
            Everything you need to make the right choice for your career
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 gap-8">
          {features.map((feature, idx) => {
            const Icon = feature.icon
            return (
              <motion.div
                key={idx}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: idx * 0.1 }}
                viewport={{ once: true }}
                className="glass-card"
              >
                <div className="flex gap-4">
                  <div className="flex-shrink-0 p-3 bg-blue-100 dark:bg-blue-900/20 rounded-lg">
                    <Icon size={24} className="text-blue-600 dark:text-blue-400" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                      {feature.title}
                    </h3>
                    <p className="text-gray-600 dark:text-gray-400">{feature.description}</p>
                  </div>
                </div>
              </motion.div>
            )
          })}
        </div>
      </section>

      {/* CTA Section */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="bg-gradient-primary rounded-2xl p-12 text-center"
        >
          <h2 className="text-4xl font-bold text-white mb-4">Ready to Find Your College?</h2>
          <p className="text-xl text-blue-100 mb-8">Start your journey with accurate predictions today</p>
          <Button variant="primary" size="lg">
            <Link to="/predictor">Get Started Now</Link>
          </Button>
        </motion.div>
      </section>
    </div>
  )
}

export default Landing
