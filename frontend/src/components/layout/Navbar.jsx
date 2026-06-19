import React, { useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { FiMenu, FiX, FiMoon, FiSun } from 'react-icons/fi'
import { useTheme } from '../../context/ThemeContext'
import { Button } from '../common/Button'

export const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false)
  const { isDark, toggleTheme } = useTheme()
  const location = useLocation()

  const navLinks = [
    { label: 'Home', href: '/' },
    { label: 'Rank Predictor', href: '/predictor' },
    { label: 'College Allocator', href: '/allocation' },
    { label: 'College Finder', href: '/colleges' },
    { label: 'Courses', href: '/courses' },
    { label: 'Analytics', href: '/analytics' },
    { label: 'About', href: '/about' },
  ]

  return (
    <nav className="sticky top-0 z-40 w-full py-4 px-4 sm:px-6 lg:px-8 bg-transparent">
      <div className="max-w-7xl mx-auto glass rounded-full px-6 py-2.5 shadow-glass border border-slate-200/40 dark:border-slate-800/40">
        <div className="flex items-center justify-between h-12">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-2.5 group">
            <div className="w-9 h-9 bg-gradient-primary rounded-xl flex items-center justify-center shadow-glow-purple group-hover:scale-105 transition-transform">
              <span className="text-white font-extrabold text-sm tracking-wide">NC</span>
            </div>
            <span className="font-extrabold text-xl tracking-tight gradient-text">NammaCET</span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center gap-1.5 bg-slate-100/50 dark:bg-slate-900/50 p-1.5 rounded-full border border-slate-200/30 dark:border-slate-800/30">
            {navLinks.map((link) => {
              const isActive = location.pathname === link.href
              return (
                <Link
                  key={link.href}
                  to={link.href}
                  className={`relative px-4 py-1.5 text-xs lg:text-sm font-semibold rounded-full transition-colors duration-300 ${
                    isActive
                      ? 'text-white'
                      : 'text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white'
                  }`}
                >
                  {isActive && (
                    <motion.div
                      layoutId="active-pill"
                      className="absolute inset-0 bg-gradient-primary rounded-full -z-10 shadow-glow-purple"
                      transition={{ type: 'spring', stiffness: 380, damping: 30 }}
                    />
                  )}
                  {link.label}
                </Link>
              )
            })}
          </div>

          {/* Right Section */}
          <div className="flex items-center gap-3">
            <button
              onClick={toggleTheme}
              className="p-2 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-full transition-colors"
            >
              {isDark ? <FiSun size={18} /> : <FiMoon size={18} />}
            </button>

            <Button variant="primary" size="sm" className="hidden sm:flex rounded-full px-5 py-2 font-bold bg-gradient-primary border-none shadow-glow-purple hover:scale-105 transition-transform" asChild>
              <Link to="/predictor">Get Started</Link>
            </Button>

            {/* Mobile Menu Button */}
            <button
              onClick={() => setIsOpen(!isOpen)}
              className="md:hidden p-2 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-full"
            >
              {isOpen ? <FiX size={20} /> : <FiMenu size={20} />}
            </button>
          </div>
        </div>

        {/* Mobile Navigation */}
        <AnimatePresence>
          {isOpen && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="md:hidden border-t border-slate-200/50 dark:border-slate-800/50 py-4 mt-2 space-y-1.5"
            >
              {navLinks.map((link) => {
                const isActive = location.pathname === link.href
                return (
                  <Link
                    key={link.href}
                    to={link.href}
                    className={`block px-4 py-2 text-sm font-semibold rounded-lg transition-colors ${
                      isActive
                        ? 'bg-primary-500/10 text-primary-500 dark:text-primary-400'
                        : 'text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800/50'
                    }`}
                    onClick={() => setIsOpen(false)}
                  >
                    {link.label}
                  </Link>
                )
              })}
              <Button variant="primary" size="sm" className="w-full mt-3 rounded-full bg-gradient-primary border-none" asChild>
                <Link to="/predictor" onClick={() => setIsOpen(false)}>Get Started</Link>
              </Button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </nav>
  )
}

export default Navbar
