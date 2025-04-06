import React, { useEffect } from 'react';

const Learning = () => {
  useEffect(() => {
    // Redirect to the standalone learning.html file
    window.location.href = '/learning.html';
  }, []);

  return (
    <div>
      <p>Redirecting to learning module...</p>
    </div>
  );
};

export default Learning;