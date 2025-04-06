import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import Navbar from '../components/Navbar';
import confetti from 'canvas-confetti';

const Test = () => {
  const [selectedAnswer, setSelectedAnswer] = useState(null);
  const [isCorrect, setIsCorrect] = useState(false);

  const handleAnswerClick = (answer) => {
    setSelectedAnswer(answer);
    
    if (answer === 'S') {
      setIsCorrect(true);
      // Trigger confetti
      confetti({
        particleCount: 150,
        spread: 70,
        origin: { y: 0.6 }
      });
    } else {
      setIsCorrect(false);
    }
  };

  return (
    <div className="min-h-screen p-6 bg-gray-50">
      <Navbar />

      <div className="mt-10 max-w-6xl mx-auto bg-white rounded-xl shadow-md p-8">
        <h2 className="text-3xl font-bold text-gray-800">Sign Language Quiz</h2>
        <p className="mt-2 text-gray-600">Questions: 15 total</p>
        
        <div className="mt-8">
          <p className="text-xl font-semibold text-gray-700">1. What letter is formed by making a fist with the thumb across the fingers?</p>
          
          <div className="grid grid-cols-2 gap-5 mt-6 max-w-xl">
            <button 
              onClick={() => handleAnswerClick('B')} 
              className={`border-2 ${selectedAnswer === 'B' ? (isCorrect ? 'border-green-500 bg-green-50' : 'border-red-500 bg-red-50') : 'border-gray-200'} rounded-lg py-4 px-6 text-lg font-medium hover:bg-gray-50 hover:-translate-y-0.5 hover:shadow-md transition-all`}
            >
              B
            </button>
            <button 
              onClick={() => handleAnswerClick('S')} 
              className={`border-2 ${selectedAnswer === 'S' ? (isCorrect ? 'border-green-500 bg-green-50' : 'border-red-500 bg-red-50') : 'border-gray-200'} rounded-lg py-4 px-6 text-lg font-medium hover:bg-gray-50 hover:-translate-y-0.5 hover:shadow-md transition-all`}
            >
              S
            </button>
            <button 
              onClick={() => handleAnswerClick('A')} 
              className={`border-2 ${selectedAnswer === 'A' ? (isCorrect ? 'border-green-500 bg-green-50' : 'border-red-500 bg-red-50') : 'border-gray-200'} rounded-lg py-4 px-6 text-lg font-medium hover:bg-gray-50 hover:-translate-y-0.5 hover:shadow-md transition-all`}
            >
              A
            </button>
            <button 
              onClick={() => handleAnswerClick('E')} 
              className={`border-2 ${selectedAnswer === 'E' ? (isCorrect ? 'border-green-500 bg-green-50' : 'border-red-500 bg-red-50') : 'border-gray-200'} rounded-lg py-4 px-6 text-lg font-medium hover:bg-gray-50 hover:-translate-y-0.5 hover:shadow-md transition-all`}
            >
              E
            </button>
          </div>
          
          {isCorrect && (
            <div className="mt-6 p-4 bg-green-100 border border-green-300 rounded-lg text-green-800 text-center">
              <p className="text-xl font-bold">🎉 Congrats! 🎉</p>
              <p>S is the correct answer. Well done!</p>
            </div>
          )}
          
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
