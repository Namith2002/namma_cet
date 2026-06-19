import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { useMutation } from '@tanstack/react-query'
import { useForm, Controller } from 'react-hook-form'
import { Select, useToast, ToastContainer } from '../components/common'
import { Button } from '../components/common/Button'
import { CollegeCard } from '../components/cards'
import { allocationService } from '../services/endpoints'
import { CATEGORIES, REGIONS } from '../constants'

const RankSliderInput = ({ label, value, onChange, min = 1, max = 250000, error }) => {
  return (
    <div className="space-y-3 p-4 bg-slate-50/70 dark:bg-slate-900/30 rounded-2xl border border-slate-200/50 dark:border-slate-800/40 shadow-sm">
      <div className="flex justify-between items-center">
        <label className="text-sm font-bold text-slate-700 dark:text-slate-300">{label}</label>
        <input
          type="number"
          min={min}
          max={max}
          value={value === '' ? '' : value}
          onChange={(e) => {
            const val = e.target.value === '' ? '' : Math.min(max, Math.max(min, Number(e.target.value) || 1))
            onChange(val)
          }}
          className="w-24 px-2 py-1 text-center font-extrabold text-sm bg-white dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700/60 rounded-lg focus:ring-2 focus:ring-primary-500 focus:outline-none dark:text-white"
        />
      </div>
      <div className="flex items-center gap-3">
        <span className="text-[10px] text-slate-400 font-bold tracking-wide">1</span>
        <input
          type="range"
          min={min}
          max={max}
          step={100}
          value={Number(value) || 1}
          onChange={(e) => onChange(Number(e.target.value))}
          className="flex-1 h-1.5 bg-slate-200 dark:bg-slate-800 rounded-lg appearance-none cursor-pointer accent-primary-500 dark:accent-primary-400"
        />
        <span className="text-[10px] text-slate-400 font-bold tracking-wide">250K</span>
      </div>
      {error && <p className="text-xs text-red-500 font-semibold mt-1">{error}</p>}
    </div>
  )
}

const Allocation = () => {
  const { control, handleSubmit, formState: { errors } } = useForm({
    defaultValues: {
      rank: 25000,
      category: 'GM',
      region: 'General',
    },
  })

  const [allocatedColleges, setAllocatedColleges] = useState(null)
  const { toasts, removeToast, error: showError, success: showSuccess } = useToast()

  const allocationMutation = useMutation({
    mutationFn: (data) => allocationService.allocateCollege(data),
    onSuccess: (result) => {
      setAllocatedColleges(result)
      showSuccess('Colleges allocated based on your preferences!')
    },
    onError: (err) => {
      showError(err.message || 'Failed to allocate colleges')
    },
  })

  const handleFormSubmit = (data) => {
    const payload = {
      rank: parseInt(data.rank),
      category: data.category,
      region: data.region,
    }
    allocationMutation.mutate(payload)
  }

  return (
    <div className="min-h-screen bg-transparent py-12 relative">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-12 text-left"
        >
          <h1 className="text-4xl font-extrabold text-slate-900 dark:text-white mb-4">
            College Allocation
          </h1>
          <p className="text-lg text-slate-600 dark:text-slate-400">
            Find eligible colleges based on your rank and preferences
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Form */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="glass-card lg:col-span-1 h-fit"
          >
            <form onSubmit={handleSubmit(handleFormSubmit)} className="space-y-5">
              <Controller
                name="rank"
                control={control}
                rules={{ required: 'Rank is required', min: { value: 1, message: 'Min 1' } }}
                render={({ field }) => (
                  <RankSliderInput
                    {...field}
                    label="Your Rank"
                    error={errors.rank?.message}
                  />
                )}
              />

              <Controller
                name="category"
                control={control}
                rules={{ required: 'Category is required' }}
                render={({ field }) => (
                  <Select
                    {...field}
                    label="Category"
                    options={CATEGORIES}
                    error={errors.category?.message}
                    required
                  />
                )}
              />

              <Controller
                name="region"
                control={control}
                rules={{ required: 'Region is required' }}
                render={({ field }) => (
                  <Select
                    {...field}
                    label="Region"
                    options={REGIONS}
                    error={errors.region?.message}
                    required
                  />
                )}
              />

              <Button
                type="submit"
                variant="primary"
                size="lg"
                className="w-full rounded-2xl py-5 font-bold bg-gradient-primary shadow-glow-purple border-none"
                isLoading={allocationMutation.isPending}
              >
                Find Colleges
              </Button>
            </form>
          </motion.div>

          {/* Results */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="lg:col-span-2 text-left"
          >
            {allocationMutation.isPending && (
              <div className="flex flex-col items-center justify-center py-24 gap-4">
                <div className="spinner w-12 h-12"></div>
                <p className="text-sm font-semibold text-slate-500 dark:text-slate-400">Searching cutoffs...</p>
              </div>
            )}

            {allocatedColleges && (
              <motion.div className="space-y-6">
                <div className="bg-primary-500/10 dark:bg-primary-500/5 rounded-2xl p-4 border border-primary-500/20 shadow-sm flex items-center gap-3">
                  <span className="w-2.5 h-2.5 rounded-full bg-primary-500 animate-pulse" />
                  <p className="text-sm font-bold text-primary-700 dark:text-primary-300">
                    Allocated <span className="underline">{allocatedColleges.length}</span> eligible options matching your preferences
                  </p>
                </div>

                <div className="grid md:grid-cols-2 gap-4">
                  {allocatedColleges.map((college, idx) => (
                    <motion.div
                      key={college.id}
                      initial={{ opacity: 0, y: 15 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: idx * 0.05 }}
                    >
                      <CollegeCard college={college} onViewDetails={() => {}} onCompare={() => {}} />
                    </motion.div>
                  ))}
                </div>
              </motion.div>
            )}

            {!allocationMutation.isPending && !allocatedColleges && (
              <div className="glass-card p-12 text-center text-slate-500 dark:text-slate-400 border border-dashed border-slate-200 dark:border-slate-800">
                <p className="font-semibold text-base mb-1">No Results Displayed</p>
                <p className="text-sm text-slate-400">Fill in the form on the left and click "Find Colleges" to run allocation.</p>
              </div>
            )}
          </motion.div>
        </div>
      </div>

      <ToastContainer toasts={toasts} removeToast={removeToast} />
    </div>
  )
}

export default Allocation
