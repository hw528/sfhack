import React from 'react';
import { Link } from 'react-router-dom';
import logoImage from '../assets/logo.png';

const Home = () => {
  return (
    <div className="min-h-screen p-6 bg-gray-50">
      <nav className="flex justify-between items-center py-4 px-6 bg-white shadow-sm rounded-lg">
        <div className="text-xl font-bold text-blue-600">SIGNLEARN</div>
        <div className="flex space-x-4">
          <Link to="/" className="bg-black text-white px-4 py-2 rounded-full hover:bg-gray-800 transition-colors">Home</Link>
          <Link to="/learning" className="bg-black text-white px-4 py-2 rounded-full hover:bg-gray-800 transition-colors">Learning 🤔</Link>
          <Link to="/test" className="bg-black text-white px-4 py-2 rounded-full hover:bg-gray-800 transition-colors">Test 🧠</Link>
        </div>
        <div className="w-8 h-8 bg-gray-200 rounded-full flex items-center justify-center">👤</div>
      </nav>

      <div className="mt-8 flex flex-col md:flex-row justify-between items-center max-w-6xl mx-auto">
        <div className="w-full md:w-1/2">
          <h1 className="text-4xl font-bold text-gray-800 mb-4">Welcome to SignLearn</h1>
          <p className="text-xl text-gray-600 mb-8">
            Learn American Sign Language (ASL) with our interactive platform. Practice signs in real-time and get instant feedback.
          </p>
          <div className="flex gap-4">
            <Link to="/learning" className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors">
              Start Learning
            </Link>
            <Link to="/test" className="bg-gray-200 text-gray-800 px-6 py-3 rounded-lg hover:bg-gray-300 transition-colors">
              Take a Test
            </Link>
          </div>
        </div>
        <div className="w-full md:w-1/2 mt-8 md:mt-0">
          <img 
            src={logoImage} 
            alt="ASL Learning Platform" 
            className="w-full rounded-lg shadow-lg"
          />
        </div>
      </div>
    </div>
  );
};

export default Home;
