import React, { createContext, useContext, useState } from 'react'

const PredictionContext = createContext()

export const PredictionProvider = ({ children }) => {
  const [predictionResult, setPredictionResult] = useState(null)
  const [allocationResults, setAllocationResults] = useState(null)
  const [comparisonData, setComparisonData] = useState(null)
  const [comparedColleges, setComparedColleges] = useState([])

  const savePrediction = (data) => {
    setPredictionResult(data)
  }

  const saveAllocation = (data) => {
    setAllocationResults(data)
  }

  const saveComparison = (data) => {
    setComparisonData(data)
  }

  const clearPrediction = () => {
    setPredictionResult(null)
  }

  const clearAllocation = () => {
    setAllocationResults(null)
  }

  const clearComparison = () => {
    setComparisonData(null)
  }

  const toggleCompareCollege = (college) => {
    setComparedColleges((prev) => {
      const exists = prev.find((c) => c.id === college.id)
      if (exists) {
        return prev.filter((c) => c.id !== college.id)
      }
      if (prev.length >= 3) {
        return prev // limit to 3 colleges
      }
      return [...prev, college]
    })
  }

  const clearComparedColleges = () => {
    setComparedColleges([])
  }

  return (
    <PredictionContext.Provider
      value={{
        predictionResult,
        savePrediction,
        clearPrediction,
        allocationResults,
        saveAllocation,
        clearAllocation,
        comparisonData,
        saveComparison,
        clearComparison,
        comparedColleges,
        toggleCompareCollege,
        clearComparedColleges,
      }}
    >
      {children}
    </PredictionContext.Provider>
  )
}

export const usePrediction = () => {
  const context = useContext(PredictionContext)
  if (!context) {
    throw new Error('usePrediction must be used within PredictionProvider')
  }
  return context
}
