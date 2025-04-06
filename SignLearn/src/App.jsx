import { Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import Learning from './pages/Learning';
import Test from './pages/Test';
import Quiz from './pages/Quiz';
import DirectCamera from './pages/DirectCamera';
import './index.css';

function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/learning" element={<Learning />} />
      <Route path="/test" element={<Test />} />
      <Route path="/quiz" element={<Quiz />} />
      <Route path="/camera-test" element={<DirectCamera />} />
    </Routes>
  );
}

export default App;