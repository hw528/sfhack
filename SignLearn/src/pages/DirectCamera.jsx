import React, { useState, useRef, useEffect } from 'react';
import { Link } from 'react-router-dom';

const DirectCamera = () => {
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [error, setError] = useState(null);
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [stream, setStream] = useState(null);

  // Effect to handle attaching stream to video when both are available
  useEffect(() => {
    if (stream && videoRef.current) {
      console.log("Effect: Attaching stream to video element");
      videoRef.current.srcObject = stream;
    }
  }, [stream, videoRef.current]);

  const startCamera = async () => {
    try {
      console.log("Starting direct camera...");
      const mediaStream = await navigator.mediaDevices.getUserMedia({ 
        video: true,
        audio: false
      });
      
      console.log("Camera stream obtained:", mediaStream);
      
      // Store the stream in state and ref
      setStream(mediaStream);
      streamRef.current = mediaStream;
      setIsCameraActive(true);
      setError(null);
      
      // Direct assignment as a backup
      if (videoRef.current) {
        console.log("Directly attaching stream to video element");
        videoRef.current.srcObject = mediaStream;
      } else {
        console.log("Video element reference not yet available - will attach in effect");
      }
    } catch (error) {
      console.error('Error starting camera:', error);
      setError(`Failed to start camera: ${error.message}`);
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setStream(null);
    setIsCameraActive(false);
  };

  return (
    <div className="min-h-screen p-6 bg-gray-50">
      <h1 className="text-2xl font-bold mb-4">Direct Camera Test</h1>
      
      {error && (
        <div className="mb-4 p-4 bg-red-100 text-red-700 rounded-lg">
          {error}
        </div>
      )}
      
      <div className="bg-black rounded-lg shadow-md overflow-hidden w-full max-w-3xl mx-auto">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-[600px] object-cover"
          style={{ transform: 'scaleX(-1)' }}
          onLoadedMetadata={() => console.log("Video element loaded metadata")}
          onPlay={() => console.log("Video started playing")}
          onError={(e) => console.error("Video error:", e)}
        />
        {!isCameraActive && (
          <div className="absolute inset-0 bg-gray-900 flex items-center justify-center text-gray-400 text-xl">
            Camera is off
          </div>
        )}
      </div>
      
      <div className="mt-4 flex gap-4 justify-center">
        <button 
          onClick={isCameraActive ? stopCamera : startCamera}
          className={`${
            isCameraActive ? 'bg-red-500 hover:bg-red-600' : 'bg-green-500 hover:bg-green-600'
          } text-white px-6 py-2 rounded-lg transition-colors`}
        >
          {isCameraActive ? 'Stop Camera' : 'Start Camera'}
        </button>
        
        <Link to="/learning" className="bg-blue-500 text-white px-6 py-2 rounded-lg hover:bg-blue-600 transition-colors">
          Back to Learning
        </Link>
      </div>
    </div>
  );
};

export default DirectCamera; 