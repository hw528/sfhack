import React from 'react';
import { Link } from 'react-router-dom';
import logoImage from '../assets/logo.png';
import Navbar from '../components/Navbar';

const Home = () => {
  return (
    <div className="min-h-screen p-6 bg-gray-50">
      <Navbar />

      <div className="mt-8 flex flex-col md:flex-row justify-between items-center max-w-6xl mx-auto">
        <div className="w-full md:w-1/2">
          <h1 className="text-4xl font-bold text-gray-800 mb-4">Welcome to SignLearn</h1>
          <p className="text-xl text-gray-600 mb-8">
            Learn American Sign Language (ASL) with our interactive platform. Practice signs in real-time and get instant feedback.
          </p>
          <div className="flex flex-wrap gap-4">
            <Link to="/learning" className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 hover:-translate-y-0.5 hover:shadow-md transition-all font-bold">
              Start Learning
            </Link>
            <Link to="/test" className="bg-gray-200 text-gray-800 px-6 py-3 rounded-lg hover:bg-gray-300 hover:-translate-y-0.5 hover:shadow-md transition-all font-bold">
              Take a Test
            </Link>
            <Link to="/quiz" className="bg-green-600 text-white px-6 py-3 rounded-lg hover:bg-green-700 hover:-translate-y-0.5 hover:shadow-md transition-all font-bold">
              Play Quiz Game
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
