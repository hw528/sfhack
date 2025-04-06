import React from 'react';
import { Link } from 'react-router-dom';

const Test = () => {
  return (
    <div className="min-h-screen p-6 bg-white">
      <nav className="flex justify-between items-center py-4 border-b">
        <div className="text-xl font-bold text-blue-600">SIGNLEARN</div>
        <div className="flex space-x-4">
          <Link to="/" className="bg-black text-white px-4 py-2 rounded-full">Home</Link>
          <Link to="/learning" className="bg-black text-white px-4 py-2 rounded-full">Learning 🤔</Link>
          <Link to="/test" className="bg-black text-white px-4 py-2 rounded-full">Test 🧠</Link>
        </div>
        <div className="w-8 h-8 bg-gray-200 rounded-full flex items-center justify-center">👤</div>
      </nav>

      <div className="mt-10">
        <h2 className="text-3xl font-bold">Sign Language Quiz (If Needed)</h2>
        <p className="mt-2">Questions no of 15</p>
        <div className="mt-6">
          <p className="text-lg font-semibold">1. What letter is formed by making a fist with the thumb across the fingers?</p>
          <div className="grid grid-cols-2 gap-4 mt-4 max-w-xl">
            <button className="border rounded-lg py-2 px-4">A )</button>
            <button className="border rounded-lg py-2 px-4">C )</button>
            <button className="border rounded-lg py-2 px-4">B )</button>
            <button className="border rounded-lg py-2 px-4">D )</button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Test;
