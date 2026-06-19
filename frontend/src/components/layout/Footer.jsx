import React from 'react'
import { Link } from 'react-router-dom'
import { FiFacebook, FiTwitter, FiLinkedin, FiInstagram } from 'react-icons/fi'

export const Footer = () => {
  const currentYear = new Date().getFullYear()

  return (
    <footer className="bg-gray-50 dark:bg-gray-900 border-t border-gray-200 dark:border-gray-800 mt-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid md:grid-cols-4 gap-8 mb-8">
          {/* Brand */}
          <div>
            <div className="flex items-center gap-2 mb-4">
              <div className="w-8 h-8 bg-gradient-primary rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-sm">NC</span>
              </div>
              <span className="font-bold text-lg gradient-text">NammaCET</span>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Karnataka CET Rank Predictor & College Allocation Platform
            </p>
          </div>

          {/* Quick Links */}
          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-4">Platform</h4>
            <ul className="space-y-2">
              <li>
                <Link to="/" className="text-sm text-gray-600 dark:text-gray-400 hover:text-blue-600">
                  Home
                </Link>
              </li>
              <li>
                <Link to="/predictor" className="text-sm text-gray-600 dark:text-gray-400 hover:text-blue-600">
                  Rank Predictor
                </Link>
              </li>
              <li>
                <Link to="/allocation" className="text-sm text-gray-600 dark:text-gray-400 hover:text-blue-600">
                  College Allocator
                </Link>
              </li>
              <li>
                <Link to="/colleges" className="text-sm text-gray-600 dark:text-gray-400 hover:text-blue-600">
                  Colleges
                </Link>
              </li>
              <li>
                <Link to="/analytics" className="text-sm text-gray-600 dark:text-gray-400 hover:text-blue-600">
                  Analytics
                </Link>
              </li>
            </ul>
          </div>

          {/* Resources */}
          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-4">Resources</h4>
            <ul className="space-y-2">
              <li>
                <Link to="/about" className="text-sm text-gray-600 dark:text-gray-400 hover:text-blue-600">
                  About Us
                </Link>
              </li>
              <li>
                <a href="#" className="text-sm text-gray-600 dark:text-gray-400 hover:text-blue-600">
                  Blog
                </a>
              </li>
              <li>
                <a href="#" className="text-sm text-gray-600 dark:text-gray-400 hover:text-blue-600">
                  FAQ
                </a>
              </li>
              <li>
                <a href="#" className="text-sm text-gray-600 dark:text-gray-400 hover:text-blue-600">
                  Contact
                </a>
              </li>
            </ul>
          </div>

          {/* Social Links */}
          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-4">Follow Us</h4>
            <div className="flex gap-4">
              <a href="#" className="text-gray-600 dark:text-gray-400 hover:text-blue-600 transition-colors">
                <FiFacebook size={20} />
              </a>
              <a href="#" className="text-gray-600 dark:text-gray-400 hover:text-blue-600 transition-colors">
                <FiTwitter size={20} />
              </a>
              <a href="#" className="text-gray-600 dark:text-gray-400 hover:text-blue-600 transition-colors">
                <FiLinkedin size={20} />
              </a>
              <a href="#" className="text-gray-600 dark:text-gray-400 hover:text-blue-600 transition-colors">
                <FiInstagram size={20} />
              </a>
            </div>
          </div>
        </div>

        {/* Bottom */}
        <div className="border-t border-gray-200 dark:border-gray-800 pt-8 flex flex-col md:flex-row items-center justify-between">
          <p className="text-sm text-gray-600 dark:text-gray-400">
            © {currentYear} NammaCET. All rights reserved.
          </p>
          <div className="flex gap-6 mt-4 md:mt-0">
            <a href="#" className="text-sm text-gray-600 dark:text-gray-400 hover:text-blue-600">
              Privacy Policy
            </a>
            <a href="#" className="text-sm text-gray-600 dark:text-gray-400 hover:text-blue-600">
              Terms of Service
            </a>
            <a href="#" className="text-sm text-gray-600 dark:text-gray-400 hover:text-blue-600">
              Disclaimer
            </a>
          </div>
        </div>
      </div>
    </footer>
  )
}
