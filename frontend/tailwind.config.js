/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#f5f3ff',
          100: '#edd8ff',
          200: '#d9b6ff',
          300: '#bd80ff',
          400: '#9d40ff',
          500: '#7c00ff', // Vivid premium purple
          600: '#6900da',
          700: '#5400ae',
          800: '#3f0083',
          900: '#2c005d',
        },
        secondary: {
          50: '#f8fafc',
          100: '#f1f5f9',
          200: '#e2e8f0',
          300: '#cbd5e1',
          400: '#94a3b8',
          500: '#64748b',
          600: '#475569',
          700: '#334155',
          800: '#1e293b',
          900: '#0f172a',
        },
        accent: {
          blue: '#3b82f6',
          cyan: '#06b6d4',
          violet: '#8b5cf6',
          purple: '#d946ef',
          pink: '#ec4899',
        },
        success: '#10b981',
        warning: '#f59e0b',
        danger: '#ef4444',
        info: '#06b6d4',
      },
      fontFamily: {
        sans: ['"Plus Jakarta Sans"', 'Inter', 'sans-serif'],
        heading: ['Outfit', 'sans-serif'],
      },
      backgroundImage: {
        'gradient-primary': 'linear-gradient(135deg, #7c00ff 0%, #3b82f6 100%)',
        'gradient-neon': 'linear-gradient(90deg, #7c00ff, #3b82f6, #06b6d4, #d946ef)',
        'gradient-dark': 'radial-gradient(ellipse at top, #1e1b4b, #0f172a, #020617)',
        'gradient-card': 'linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.01) 100%)',
      },
      backdropFilter: {
        'glass': 'blur(16px)',
      },
      boxShadow: {
        'glass': '0 8px 32px 0 rgba(124, 0, 255, 0.15)',
        'glass-hover': '0 12px 48px 0 rgba(124, 0, 255, 0.3)',
        'card': '0 10px 30px -5px rgba(0, 0, 0, 0.2)',
        'glow-blue': '0 0 20px 2px rgba(59, 130, 246, 0.4)',
        'glow-purple': '0 0 20px 2px rgba(124, 0, 255, 0.4)',
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'slide-up': 'slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1)',
        'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'float': 'float 6s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-10px)' },
        },
      },
    },
  },
  darkMode: 'class',
  plugins: [],
}
