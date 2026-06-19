import React from 'react'
import { useForm, Controller } from 'react-hook-form'
import { motion } from 'framer-motion'
import { Select } from '../common/Select'
import { Button } from '../common/Button'
import { CATEGORIES, REGIONS } from '../../constants'

const SliderInput = ({ label, value, onChange, min = 0, max = 100, error }) => {
  return (
    <div className="space-y-3 p-4 bg-slate-50/70 dark:bg-slate-900/30 rounded-2xl border border-slate-200/50 dark:border-slate-800/40 shadow-sm hover:shadow-md transition-shadow duration-300">
      <div className="flex justify-between items-center">
        <label className="text-xs lg:text-sm font-bold text-slate-700 dark:text-slate-300">{label}</label>
        <input
          type="number"
          min={min}
          max={max}
          value={value === '' ? '' : value}
          onChange={(e) => {
            const val = e.target.value === '' ? '' : Math.min(max, Math.max(min, Number(e.target.value) || 0))
            onChange(val)
          }}
          className="w-16 px-2 py-1 text-center font-extrabold text-sm bg-white dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700/60 rounded-lg focus:ring-2 focus:ring-primary-500 focus:outline-none dark:text-white"
        />
      </div>
      <div className="flex items-center gap-3">
        <span className="text-[10px] text-slate-400 font-bold tracking-wide">{min}</span>
        <input
          type="range"
          min={min}
          max={max}
          value={Number(value) || 0}
          onChange={(e) => onChange(Number(e.target.value))}
          className="flex-1 h-1.5 bg-slate-200 dark:bg-slate-800 rounded-lg appearance-none cursor-pointer accent-primary-500 dark:accent-primary-400"
        />
        <span className="text-[10px] text-slate-400 font-bold tracking-wide">{max}</span>
      </div>
      {error && <p className="text-xs text-red-500 font-semibold mt-1">{error}</p>}
    </div>
  )
}

export const RankPredictorForm = ({ onSubmit, isLoading = false }) => {
  const { control, handleSubmit, formState: { errors } } = useForm({
    defaultValues: {
      physics_kcet: 30,
      chemistry_kcet: 30,
      mathematics_kcet: 30,
      biology_kcet: 0,
      physics_theory: 75,
      chemistry_theory: 75,
      mathematics_theory: 75,
      biology_theory: 0,
      category: 'GM',
      region: 'General',
    },
  })

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
      },
    },
  }

  const itemVariants = {
    hidden: { opacity: 0, y: 15 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { type: 'spring', stiffness: 300, damping: 24 },
    },
  }

  return (
    <motion.form
      onSubmit={handleSubmit(onSubmit)}
      className="space-y-6 text-left"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      <div className="grid md:grid-cols-2 gap-6">
        {/* KCET Marks Section */}
        <motion.div variants={itemVariants} className="space-y-4">
          <div className="flex items-center gap-2 pb-2 border-b border-slate-200/50 dark:border-slate-800/50">
            <span className="w-2.5 h-2.5 rounded-full bg-primary-500 shadow-glow-purple" />
            <h3 className="text-base font-extrabold text-slate-900 dark:text-white">KCET Marks (Out of 60)</h3>
          </div>
          <div className="space-y-3">
            <Controller
              name="physics_kcet"
              control={control}
              rules={{ required: 'Required', min: { value: 0, message: 'Min 0' }, max: { value: 60, message: 'Max 60' } }}
              render={({ field }) => (
                <SliderInput {...field} label="Physics" min={0} max={60} error={errors.physics_kcet?.message} />
              )}
            />
            <Controller
              name="chemistry_kcet"
              control={control}
              rules={{ required: 'Required', min: { value: 0, message: 'Min 0' }, max: { value: 60, message: 'Max 60' } }}
              render={({ field }) => (
                <SliderInput {...field} label="Chemistry" min={0} max={60} error={errors.chemistry_kcet?.message} />
              )}
            />
            <Controller
              name="mathematics_kcet"
              control={control}
              rules={{ required: 'Required', min: { value: 0, message: 'Min 0' }, max: { value: 60, message: 'Max 60' } }}
              render={({ field }) => (
                <SliderInput {...field} label="Mathematics" min={0} max={60} error={errors.mathematics_kcet?.message} />
              )}
            />
            <Controller
              name="biology_kcet"
              control={control}
              rules={{ min: { value: 0, message: 'Min 0' }, max: { value: 60, message: 'Max 60' } }}
              render={({ field }) => (
                <SliderInput {...field} label="Biology" min={0} max={60} error={errors.biology_kcet?.message} />
              )}
            />
          </div>
        </motion.div>

        {/* Theory Marks Section */}
        <motion.div variants={itemVariants} className="space-y-4">
          <div className="flex items-center gap-2 pb-2 border-b border-slate-200/50 dark:border-slate-800/50">
            <span className="w-2.5 h-2.5 rounded-full bg-accent-blue shadow-glow-blue" />
            <h3 className="text-base font-extrabold text-slate-900 dark:text-white">Board Theory Marks (Out of 100)</h3>
          </div>
          <div className="space-y-3">
            <Controller
              name="physics_theory"
              control={control}
              rules={{ required: 'Required', min: { value: 0, message: 'Min 0' }, max: { value: 100, message: 'Max 100' } }}
              render={({ field }) => (
                <SliderInput {...field} label="Physics" min={0} max={100} error={errors.physics_theory?.message} />
              )}
            />
            <Controller
              name="chemistry_theory"
              control={control}
              rules={{ required: 'Required', min: { value: 0, message: 'Min 0' }, max: { value: 100, message: 'Max 100' } }}
              render={({ field }) => (
                <SliderInput {...field} label="Chemistry" min={0} max={100} error={errors.chemistry_theory?.message} />
              )}
            />
            <Controller
              name="mathematics_theory"
              control={control}
              rules={{ required: 'Required', min: { value: 0, message: 'Min 0' }, max: { value: 100, message: 'Max 100' } }}
              render={({ field }) => (
                <SliderInput {...field} label="Mathematics" min={0} max={100} error={errors.mathematics_theory?.message} />
              )}
            />
            <Controller
              name="biology_theory"
              control={control}
              rules={{ min: { value: 0, message: 'Min 0' }, max: { value: 100, message: 'Max 100' } }}
              render={({ field }) => (
                <SliderInput {...field} label="Biology" min={0} max={100} error={errors.biology_theory?.message} />
              )}
            />
          </div>
        </motion.div>
      </div>

      {/* Category & Region Section */}
      <motion.div variants={itemVariants} className="grid md:grid-cols-2 gap-6 pt-4 border-t border-slate-200/50 dark:border-slate-800/50">
        <Controller
          name="category"
          control={control}
          rules={{ required: 'Category is required' }}
          render={({ field }) => (
            <Select
              {...field}
              label="Category Merit"
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
              label="Quota Region"
              options={REGIONS}
              error={errors.region?.message}
              required
            />
          )}
        />
      </motion.div>

      {/* Submit Button */}
      <motion.div variants={itemVariants} className="pt-4">
        <Button
          type="submit"
          variant="primary"
          size="lg"
          className="w-full rounded-2xl py-6 font-extrabold text-sm tracking-wide bg-gradient-primary shadow-glow-purple border-none hover:scale-[1.01] hover:shadow-glass-hover transition-all"
          isLoading={isLoading}
        >
          {isLoading ? 'Forecasting Rank...' : 'Forecast My KCET Rank'}
        </Button>
      </motion.div>
    </motion.form>
  )
}
