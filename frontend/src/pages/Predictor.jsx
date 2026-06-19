import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { useQuery, useMutation } from '@tanstack/react-query'
import { RankPredictorForm } from '../components/forms/RankPredictorForm'
import { Loader, useToast, ToastContainer } from '../components/common'
import { predictionService } from '../services/endpoints'
import { usePrediction } from '../context/PredictionContext'
import { formatRank, formatScore, calculatePercentage } from '../utils/helpers'
import { SimpleBarChart, SimpleLineChart } from '../components/charts'
import { CollegeCard } from '../components/cards'
import { FiAward } from 'react-icons/fi'

const Predictor = () => {
  const { savePrediction } = usePrediction()
  const { toasts, removeToast, error: showError } = useToast()
  const [prediction, setPrediction] = useState(null)

  const predictMutation = useMutation({
    mutationFn: (data) => predictionService.predictRank(data),
    onSuccess: (result) => {
      setPrediction(result)
      savePrediction(result)
    },
    onError: (err) => {
      showError(err.message || 'Failed to predict rank')
    },
  })

  const handleFormSubmit = (data) => {
    const payload = {
      kcet_physics: parseFloat(data.physics_kcet),
      kcet_chemistry: parseFloat(data.chemistry_kcet),
      kcet_mathematics: parseFloat(data.mathematics_kcet),
      kcet_biology: parseFloat(data.biology_kcet) || 0,
      theory_physics: parseFloat(data.physics_theory),
      theory_chemistry: parseFloat(data.chemistry_theory),
      theory_mathematics: parseFloat(data.mathematics_theory),
      theory_biology: parseFloat(data.biology_theory) || 0,
      category: data.category,
      region: data.region,
    }
    predictMutation.mutate(payload)
  }

  return (
    <div className="min-h-screen bg-white dark:bg-gray-900">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">KCET Rank Predictor</h1>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            Enter your marks to get an accurate rank prediction based on ML models
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-8 max-w-5xl mx-auto">
          {/* Form Section */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="glass-card"
          >
            <RankPredictorForm
              onSubmit={handleFormSubmit}
              isLoading={predictMutation.isPending}
            />
          </motion.div>

          {/* Results Section */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="space-y-6"
          >
            {predictMutation.isPending && (
              <Loader size="lg" />
            )}

            {prediction && (
              <>
                {/* Predicted Rank Card */}
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="glass-card bg-gradient-primary text-white relative overflow-hidden shadow-glow-purple border-none p-8"
                >
                  <div className="absolute right-0 bottom-0 translate-x-6 translate-y-6 opacity-10">
                    <FiAward size={180} />
                  </div>
                  <div className="relative z-10">
                    <span className="inline-block px-2.5 py-0.5 rounded-full bg-white/20 text-xs font-bold uppercase tracking-wider mb-3">
                      Predicted Result
                    </span>
                    <p className="text-sm opacity-90 mb-1 font-semibold">Estimated KCET Rank</p>
                    <p className="text-5xl lg:text-6xl font-extrabold tracking-tight mb-2">
                      {formatRank(prediction.predicted_rank)}
                    </p>
                    <p className="text-xs opacity-75">
                      Analyzed via Gradient Boosting & Random Forest models for Category: <strong className="opacity-100">{prediction.category_code}</strong> ({prediction.category_type})
                    </p>
                  </div>
                </motion.div>

                {/* Statistics */}
                <div className="grid grid-cols-2 gap-4 mb-6">
                  {[
                    { label: 'KCET Total', value: `${formatScore(prediction.kcet_total)}/240`, desc: 'Entrance score' },
                    { label: 'Board Theory', value: `${formatScore(prediction.theory_total)}/400`, desc: 'School theory' },
                    { label: 'Combined Score', value: `${formatScore(prediction.combined_score)}/640`, desc: 'Aggregated total' },
                    { label: 'KCET Percentage', value: `${(prediction.kcet_total / 2.4).toFixed(1)}%`, desc: 'Exam percentile' },
                  ].map((stat, idx) => (
                    <motion.div
                      key={idx}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: idx * 0.08 }}
                      className="glass-card p-4 flex flex-col justify-between border-slate-200/50 dark:border-slate-800/40"
                    >
                      <div>
                        <p className="text-xs font-bold text-slate-500 dark:text-slate-400">{stat.label}</p>
                        <p className="text-2xl font-extrabold text-slate-900 dark:text-white mt-1">{stat.value}</p>
                      </div>
                      <p className="text-[10px] text-slate-400 font-medium mt-2">{stat.desc}</p>
                    </motion.div>
                  ))}
                </div>

                {/* Recommended Colleges */}
                {prediction.eligible_colleges && prediction.eligible_colleges.length > 0 && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.4 }}
                    className="space-y-4 pt-4"
                  >
                    <h3 className="text-xl font-bold text-gray-900 dark:text-white">
                      Recommended/Eligible Colleges
                    </h3>
                    <div className="space-y-3">
                      {prediction.eligible_colleges.map((college, idx) => (
                        <motion.div
                          key={college.id || idx}
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: 0.4 + idx * 0.05 }}
                        >
                          <CollegeCard
                            college={college}
                            onViewDetails={() => {}}
                            onCompare={() => {}}
                            userRank={prediction.predicted_rank}
                            category={prediction.category_code}
                          />
                        </motion.div>
                      ))}
                    </div>
                  </motion.div>
                )}
              </>
            )}
          </motion.div>
        </div>
      </div>

      <ToastContainer toasts={toasts} removeToast={removeToast} />
    </div>
  )
}

export default Predictor
