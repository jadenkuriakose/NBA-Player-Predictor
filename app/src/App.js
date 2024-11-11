import React, { useState, useRef } from 'react';
import { FaMicrophone } from 'react-icons/fa';
import axios from 'axios';
import './App.css';

function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [query, setQuery] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

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

      recognitionRef.current.onstart = () => {
        setIsRecording(true);
      };

      recognitionRef.current.onresult = (event) => {
        const transcript = Array.from(event.results)
          .map(result => result[0])
          .map(result => result.transcript)
          .join('');

        setQuery(transcript);
      };

      recognitionRef.current.onerror = () => {
        setIsRecording(false);
      };

      recognitionRef.current.onend = () => {
        setIsRecording(false);
      };

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

  const handlePredict = async () => {
    if (!query.trim()) {
      alert('Please enter a player name.');
      return;
    }

    setLoading(true);

    try {
      const response = await axios.post('http://127.0.0.1:8080/predict', {
        player_name: query
      });
      
      setPrediction(response.data);
    } catch (error) {
      alert('Error fetching prediction.');
    } finally {
      setLoading(false);
    }
  };

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

        <button className="predict" onClick={handlePredict} disabled={loading}>
          {loading ? 'Loading...' : 'Predict!'}
        </button>

        {prediction && (
          <div className="prediction-result">
            <h2>Prediction Result</h2>
            <div className="prediction-stats">
              <p><strong>Predicted Points:</strong> {prediction.PTS}</p>
              <p><strong>Predicted Assists:</strong> {prediction.AST}</p>
              <p><strong>Predicted Rebounds:</strong> {prediction.TRB}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;