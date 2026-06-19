import React from 'react'
import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import { FiArrowRight, FiTrendingUp, FiTarget, FiBook, FiBarChart2, FiUsers, FiAward } from 'react-icons/fi'
import { Button } from '../components/common/Button'
import { StatCard } from '../components/cards'

const Landing = () => {
  const features = [
    {
      icon: FiTarget,
      title: 'Accurate Predictions',
      description: 'State-of-the-art ML-powered rank forecasting using extensive multi-year historical datasets.',
    },
    {
      icon: FiBook,
      title: 'College & Course Allocator',
      description: 'Explore matching colleges based on your rank with smart category and region adjustments.',
    },
    {
      icon: FiBarChart2,
      title: 'Analytics Dashboard',
      description: 'Unlock insights into cutoffs, distributions, and popularity with premium visual components.',
    },
    {
      icon: FiAward,
      title: 'Side-by-Side Comparison',
      description: 'Compare multiple colleges directly on key metrics to make confident admission decisions.',
    },
  ]

  const stats = [
    { title: 'Total Colleges', value: '260+', icon: FiUsers, color: 'blue' },
    { title: 'Total Courses', value: '500+', icon: FiBook, color: 'purple' },
    { title: 'Success Rate', value: '95%', icon: FiTrendingUp, color: 'green' },
    { title: 'Students Guided', value: '12K+', icon: FiAward, color: 'orange' },
  ]

  return (
    <div className="min-h-screen bg-transparent relative overflow-hidden">
      {/* Decorative Grid Backdrop */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808008_1px,transparent_1px),linear-gradient(to_bottom,#80808008_1px,transparent_1px)] bg-[size:32px_32px] pointer-events-none" />

      {/* Hero Section */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-16 pb-24 md:pt-28 md:pb-32 relative">
        <div className="grid lg:grid-cols-12 gap-12 items-center">
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, cubicBezier: [0.16, 1, 0.3, 1] }}
            className="lg:col-span-6 text-left"
          >
            <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-primary-500/10 text-primary-500 dark:text-primary-400 text-xs font-semibold tracking-wide mb-6 border border-primary-500/20">
              <span className="w-1.5 h-1.5 rounded-full bg-primary-500 animate-pulse" />
              Advanced ML Rank Prediction
            </div>
            <h1 className="text-4xl sm:text-5xl md:text-6xl font-extrabold text-slate-900 dark:text-white leading-tight tracking-tight mb-6">
              Predict Your <span className="gradient-text">KCET Rank</span> & Get Allocated.
            </h1>
            <p className="text-base sm:text-lg text-slate-600 dark:text-slate-400 mb-8 max-w-xl leading-relaxed">
              Use our high-precision predictive models, explore cutoff trends, and find the perfect college matching your scores, preferences, and category.
            </p>
            <div className="flex flex-wrap gap-4">
              <Button variant="primary" size="lg" className="rounded-full shadow-glow-purple px-7 py-6 text-sm font-bold flex items-center gap-2 bg-gradient-primary border-none hover:scale-105 transition-transform" asChild>
                <Link to="/predictor">
                  Predict Your Rank
                  <FiArrowRight size={18} />
                </Link>
              </Button>
              <Button variant="outline" size="lg" className="rounded-full px-7 py-6 text-sm font-bold border-slate-200/60 dark:border-slate-800/60 hover:bg-slate-100 dark:hover:bg-slate-800/40" asChild>
                <Link to="/colleges">Explore Colleges</Link>
              </Button>
            </div>
          </motion.div>

          {/* Interactive GUI Dashboard Mockup */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95, x: 30 }}
            animate={{ opacity: 1, scale: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.1, cubicBezier: [0.16, 1, 0.3, 1] }}
            className="lg:col-span-6 relative flex justify-center"
          >
            {/* Background Glows */}
            <div className="absolute w-72 h-72 bg-primary-500/20 dark:bg-primary-500/10 blur-[100px] -z-10 rounded-full" />
            <div className="absolute w-72 h-72 bg-accent-blue/20 dark:bg-accent-blue/10 blur-[100px] -z-10 rounded-full bottom-0 right-0" />

            <div className="w-full max-w-lg p-2.5 rounded-2xl glass border border-slate-200/40 dark:border-slate-800/40 shadow-glass animate-float relative overflow-hidden group">
              <div className="absolute inset-0 bg-gradient-to-tr from-white/10 to-transparent dark:from-transparent opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none duration-500" />
              <img
                src="/dashboard_preview.png"
                alt="NammaCET Rank Predictor GUI Mockup"
                className="w-full h-auto rounded-xl object-cover"
              />
            </div>
          </motion.div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid sm:grid-cols-2 md:grid-cols-4 gap-6">
          {stats.map((stat, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 15 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: idx * 0.08 }}
            >
              <StatCard {...stat} />
            </motion.div>
          ))}
        </div>
      </section>

      {/* Features Section */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 relative">
        <div className="absolute w-72 h-72 bg-accent-violet/10 blur-[120px] top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 pointer-events-none rounded-full" />

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl sm:text-4xl font-extrabold text-slate-900 dark:text-white mb-4">
            Why Choose NammaCET?
          </h2>
          <p className="text-base sm:text-lg text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
            Everything you need to map out your KCET results and discover your best college options.
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
                transition={{ delay: idx * 0.08 }}
                viewport={{ once: true }}
                className="glass-card flex gap-5 items-start relative group"
              >
                <div className="flex-shrink-0 p-3.5 bg-primary-500/10 text-primary-500 dark:text-primary-400 rounded-xl group-hover:scale-110 transition-transform duration-300">
                  <Icon size={22} />
                </div>
                <div>
                  <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-2 group-hover:text-primary-500 dark:group-hover:text-primary-400 transition-colors duration-300">
                    {feature.title}
                  </h3>
                  <p className="text-sm sm:text-base text-slate-600 dark:text-slate-400 leading-relaxed">
                    {feature.description}
                  </p>
                </div>
              </motion.div>
            )
          })}
        </div>
      </section>

      {/* How It Works Section */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 bg-slate-100/40 dark:bg-slate-900/10 rounded-3xl border border-slate-200/30 dark:border-slate-800/30">
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl sm:text-4xl font-extrabold text-slate-900 dark:text-white mb-4">
            How It Works
          </h2>
          <p className="text-slate-600 dark:text-slate-400 text-base max-w-md mx-auto">
            Get accurate reports and college allocations in three simple steps.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-3 gap-8 relative">
          {[
            { step: 1, title: 'Enter Your Scores', desc: 'Input your KCET subjects and board theory marks.' },
            { step: 2, title: 'Select Preferences', desc: 'Configure your category codes, regions, and courses.' },
            { step: 3, title: 'Explore & Compare', desc: 'Browse allocated recommendations and compare options side-by-side.' },
          ].map((item, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.1 }}
              viewport={{ once: true }}
              className="text-center relative group px-4"
            >
              <div className="w-14 h-14 mx-auto mb-5 bg-gradient-primary rounded-2xl flex items-center justify-center text-white text-xl font-extrabold shadow-glow-purple group-hover:rotate-6 transition-transform">
                {item.step}
              </div>
              <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-2">{item.title}</h3>
              <p className="text-sm text-slate-600 dark:text-slate-400 leading-relaxed">{item.desc}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* CTA Section */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="bg-gradient-primary rounded-3xl p-12 text-center shadow-glow-purple relative overflow-hidden"
        >
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_bottom_right,rgba(255,255,255,0.1),transparent)] pointer-events-none" />
          <h2 className="text-3xl sm:text-4xl font-extrabold text-white mb-4">
            Ready to Find Your College?
          </h2>
          <p className="text-lg text-primary-100 mb-8 max-w-lg mx-auto leading-relaxed">
            Foresee your ranks, explore historical cutoffs, and customize your allocations.
          </p>
          <Button variant="primary" size="lg" className="rounded-full px-8 py-6 text-sm font-bold bg-white text-primary-600 hover:bg-slate-100 hover:scale-105 border-none shadow-card transition-transform" asChild>
            <Link to="/predictor">Get Started Now</Link>
          </Button>
        </motion.div>
      </section>
    </div>
  )
}

export default Landing
