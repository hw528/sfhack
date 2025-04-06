import React from 'react';
import { Link, useLocation } from 'react-router-dom';

const Navbar = () => {
  const location = useLocation();
  
  return (
    <div className="flex justify-between items-center p-5 bg-white shadow-sm rounded-xl mb-8">
      <div className="text-2xl font-bold text-blue-600">SIGNLEARN</div>
      <div className="flex space-x-4">
        <Link to="/" className={`px-6 py-3 rounded-full text-white font-medium text-base transition-all duration-200 transform hover:-translate-y-0.5 hover:shadow-md ${location.pathname === '/' ? 'bg-blue-600 hover:bg-blue-700' : 'bg-black hover:bg-gray-800'}`}>Home</Link>
        <Link to="/learning" className={`px-6 py-3 rounded-full text-white font-medium text-base transition-all duration-200 transform hover:-translate-y-0.5 hover:shadow-md ${location.pathname === '/learning' ? 'bg-blue-600 hover:bg-blue-700' : 'bg-black hover:bg-gray-800'}`}>Learning</Link>
        <Link to="/quiz" className={`px-6 py-3 rounded-full text-white font-medium text-base transition-all duration-200 transform hover:-translate-y-0.5 hover:shadow-md ${location.pathname === '/quiz' ? 'bg-blue-600 hover:bg-blue-700' : 'bg-black hover:bg-gray-800'}`}>Quiz</Link>
        <Link to="/test" className={`px-6 py-3 rounded-full text-white font-medium text-base transition-all duration-200 transform hover:-translate-y-0.5 hover:shadow-md ${location.pathname === '/test' ? 'bg-blue-600 hover:bg-blue-700' : 'bg-black hover:bg-gray-800'}`}>Test</Link>
      </div>
      <div className="w-[42px] h-[42px] bg-gray-200 rounded-full flex items-center justify-center text-lg">👤</div>
    </div>
  );
};

export default Navbar; 