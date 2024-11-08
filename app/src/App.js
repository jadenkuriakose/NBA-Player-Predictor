import React, { useState, useRef } from 'react';
import { FaSearch, FaMicrophone } from 'react-icons/fa';
import './App.css';

function NbaSearchApp() {
  const [isRecording, setIsRecording] = useState(false);
  const [query, setQuery] = useState('');
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

      recognitionRef.current.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
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

  return (
    <div className="App">
      <h1>NBA Search</h1>

      <div className="search-box">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search NBA teams or players"
          className="search-input"
        />

        <button className="mic-btn" onClick={toggleRecording}>
          {isRecording ? <FaMicrophone color="red" /> : <FaMicrophone />}
        </button>
      </div>

      <button className="predict">
          Predict!
      </button>

    </div>
  );
}

export default NbaSearchApp;
