import React from 'react'
import { motion } from 'framer-motion'
import { FiCheckCircle, FiGitBranch, FiCpu, FiZap } from 'react-icons/fi'

const About = () => {
  const features = [
    {
      icon: FiGitBranch,
      title: 'Gradient Boosting',
      description: 'Ensemble learning for accurate predictions',
    },
    {
      icon: FiCheckCircle,
      title: 'Random Forest',
      description: 'Multiple decision trees for robust results',
    },
    {
      icon: FiCpu,
      title: 'Linear Regression',
      description: 'Statistical analysis for trend detection',
    },
    {
      icon: FiZap,
      title: 'Real-time Updates',
      description: 'Up-to-date cutoff and admission data',
    },
  ]

  const techStack = [
    { category: 'Frontend', tech: 'React 18, Vite, Tailwind CSS, Framer Motion' },
    { category: 'Backend', tech: 'Python, Flask, Machine Learning Models' },
    { category: 'Database', tech: 'PostgreSQL, Redis Cache' },
    { category: 'Deployment', tech: 'Docker, AWS, CI/CD Pipelines' },
  ]

  return (
    <div className="min-h-screen bg-white dark:bg-gray-900 py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Hero Section */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-16"
        >
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">About NammaCET</h1>
          <p className="text-xl text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
            Empowering Karnataka students to make informed decisions about their future education
            through accurate rank predictions and comprehensive college information.
          </p>
        </motion.div>

        {/* Mission Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="glass-card mb-12 p-8"
        >
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">Our Mission</h2>
          <p className="text-lg text-gray-600 dark:text-gray-400 leading-relaxed">
            NammaCET is designed to simplify the complex KCET (Karnataka Common Entrance Test)
            rank prediction and college allocation process. Using advanced machine learning algorithms
            trained on historical KCET data, we provide accurate rank predictions that help students
            plan their future and choose the right colleges. Our platform combines technology with
            user-friendly design to make college selection stress-free and data-driven.
          </p>
        </motion.div>

        {/* How Prediction Works */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="mb-16"
        >
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-8 text-center">
            How Prediction Works
          </h2>

          <div className="grid md:grid-cols-4 gap-6">
            {[
              {
                step: 1,
                title: 'Data Collection',
                desc: 'Analyzing historical KCET marks and rank data',
              },
              {
                step: 2,
                title: 'ML Training',
                desc: 'Training algorithms with preprocessed data',
              },
              {
                step: 3,
                title: 'Model Validation',
                desc: 'Validating accuracy on test datasets',
              },
              {
                step: 4,
                title: 'Predictions',
                desc: 'Real-time rank predictions for students',
              },
            ].map((item, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: idx * 0.1 }}
                viewport={{ once: true }}
                className="glass-card"
              >
                <div className="text-center">
                  <div className="w-12 h-12 mx-auto mb-3 bg-blue-600 rounded-full flex items-center justify-center text-white text-xl font-bold">
                    {item.step}
                  </div>
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                    {item.title}
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">{item.desc}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* ML Models */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="mb-16"
        >
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-8 text-center">
            Machine Learning Models
          </h2>

          <div className="grid md:grid-cols-2 gap-6">
            {features.map((feature, idx) => {
              const Icon = feature.icon
              return (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, x: idx % 2 === 0 ? -20 : 20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.1 }}
                  viewport={{ once: true }}
                  className="glass-card"
                >
                  <div className="flex gap-4">
                    <div className="p-3 bg-blue-100 dark:bg-blue-900/20 rounded-lg h-fit">
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
        </motion.div>

        {/* Tech Stack */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="mb-16"
        >
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-8 text-center">
            Technology Stack
          </h2>

          <div className="grid md:grid-cols-2 gap-6">
            {techStack.map((item, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: idx * 0.1 }}
                viewport={{ once: true }}
                className="glass-card p-6"
              >
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                  {item.category}
                </h3>
                <p className="text-gray-600 dark:text-gray-400">{item.tech}</p>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Contact Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="bg-gradient-primary rounded-2xl p-12 text-center"
        >
          <h2 className="text-3xl font-bold text-white mb-4">Have Questions?</h2>
          <p className="text-blue-100 mb-6">Reach out to our team for more information</p>
          <a
            href="mailto:info@nammacet.com"
            className="inline-block bg-white text-blue-600 font-semibold px-6 py-3 rounded-lg hover:bg-blue-50 transition-colors"
          >
            Contact Us
          </a>
        </motion.div>
      </div>
    </div>
  )
}

export default About
