import React, { useState, useRef, useCallback } from 'react';
import { FaMicrophone } from 'react-icons/fa';
import axios from 'axios';
import './App.css';

function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null); // Combined state for prediction and explanation
  const recognitionRef = useRef(null);

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  const startRecording = () => {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = window.webkitSpeechRecognition || window.SpeechRecognition;
      recognitionRef.current = new SpeechRecognition();

      recognitionRef.current.continuous = true;
      recognitionRef.current.interimResults = true;

      recognitionRef.current.onstart = () => setIsRecording(true);

      recognitionRef.current.onresult = (event) => {
        const transcript = Array.from(event.results)
          .map(result => result[0])
          .map(result => result.transcript)
          .join('');
        setQuery(transcript);
      };

      recognitionRef.current.onerror = () => setIsRecording(false);

      recognitionRef.current.onend = () => setIsRecording(false);

      recognitionRef.current.start();
    } else {
      alert('Speech recognition is not supported in your browser.');
    }
  };

  const stopRecording = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      setIsRecording(false);
    }
  };

  const handlePredictionAndExplanation = useCallback(async () => {
    if (!query.trim()) {
      alert('Please enter a player name.');
      return;
    }

    setLoading(true);

    try {
      // Fetch both prediction and explanation at the same time
      const [predictionResponse, explanationResponse] = await Promise.all([
        axios.post('http://127.0.0.1:8080/predict', { player_name: query }),
        axios.post('http://127.0.0.1:8080/explanation', { player_name: query })
      ]);

      if (predictionResponse.data && explanationResponse.data) {
        setResult({
          prediction: predictionResponse.data,
          explanation: explanationResponse.data.explanation || 'No explanation available.',
        });
      } else {
        alert('Prediction or explanation data not available.');
      }
    } catch (error) {
      console.error('Error fetching prediction or explanation:', error);
      alert('Error fetching prediction or explanation.');
    } finally {
      setLoading(false);
    }
  }, [query]);

  return (
    <div className="App">
      <div className="content-wrapper">
        <h1>NBA Player Stat Prediction</h1>

        <div className="search-box">
          <div className="search-input-wrapper">
            <span className="search-icon">🔍</span>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search NBA players"
              className="search-input"
            />
            <button
              className={`mic-btn ${isRecording ? 'recording' : ''}`}
              onClick={toggleRecording}
            >
              <FaMicrophone size={16} />
            </button>
          </div>
        </div>

        <button className="predict" onClick={handlePredictionAndExplanation} disabled={loading}>
          {loading ? 'Loading...' : 'Predict!'}
        </button>

        {result && (
          <div className="prediction-result">
            <h2>Prediction Result</h2>
            <div className="prediction-stats">
              <p><strong>Predicted Points:</strong> {result.prediction?.PTS}</p>
              <p><strong>Predicted Assists:</strong> {result.prediction?.AST}</p>
              <p><strong>Predicted Rebounds:</strong> {result.prediction?.TRB}</p>
            </div>

            {result.explanation && (
              <div className="prediction-explanation">
                <h3>Explanation</h3>
                <p>{result.explanation}</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
