import React from 'react';
import { Link } from 'react-router-dom';
import Navbar from '../components/Navbar';

const Test = () => {
  return (
    <div className="min-h-screen p-6 bg-gray-50">
      <Navbar />

      <div className="mt-10 max-w-6xl mx-auto bg-white rounded-xl shadow-md p-8">
        <h2 className="text-3xl font-bold text-gray-800">Sign Language Quiz</h2>
        <p className="mt-2 text-gray-600">Questions: 15 total</p>
        
        <div className="mt-8">
          <p className="text-xl font-semibold text-gray-700">1. What letter is formed by making a fist with the thumb across the fingers?</p>
          
          <div className="grid grid-cols-2 gap-5 mt-6 max-w-xl">
            <button className="border-2 border-gray-200 rounded-lg py-4 px-6 text-lg font-medium hover:bg-gray-50 hover:-translate-y-0.5 hover:shadow-md transition-all">A</button>
            <button className="border-2 border-gray-200 rounded-lg py-4 px-6 text-lg font-medium hover:bg-gray-50 hover:-translate-y-0.5 hover:shadow-md transition-all">C</button>
            <button className="border-2 border-gray-200 rounded-lg py-4 px-6 text-lg font-medium hover:bg-gray-50 hover:-translate-y-0.5 hover:shadow-md transition-all">B</button>
            <button className="border-2 border-gray-200 rounded-lg py-4 px-6 text-lg font-medium hover:bg-gray-50 hover:-translate-y-0.5 hover:shadow-md transition-all">D</button>
          </div>
          
          <div className="mt-8 flex justify-between">
            <button className="bg-gray-200 text-gray-800 px-6 py-3 rounded-lg hover:bg-gray-300 hover:-translate-y-0.5 hover:shadow-md transition-all font-bold disabled:opacity-50 disabled:cursor-not-allowed" disabled>
              Previous
            </button>
            <button className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 hover:-translate-y-0.5 hover:shadow-md transition-all font-bold">
              Next
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Test;
