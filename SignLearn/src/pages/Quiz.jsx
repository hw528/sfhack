import React, { useEffect } from 'react';

const Quiz = () => {
  useEffect(() => {
    // Redirect to the standalone quiz.html file
    window.location.href = '/quiz.html';
  }, []);

  return (
    <div>
      <p>Redirecting to quiz game...</p>
    </div>
  );
};

export default Quiz; 