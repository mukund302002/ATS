// src/components/PromptInput.jsx
import React, { useState } from 'react';
import './PromptInput.css';

const PromptInput = () => {
  const [category, setcategory] = useState('');
  const [prompt, setPrompt] = useState('');
  const [numCandidates, setNumCandidates] = useState('');
  const [output, setOutput] = useState({});

  const handleSubmit = async (e) => {
    e.preventDefault();
    const response = await fetch('http://localhost:5000/api/prompt', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ category, prompt, numCandidates }),
    });
    const data = await response.json();
    setOutput(data);
  };

  return (
    <div style={{ width: '310%', display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
      <div className="container">
        <div className="input-group">
          <input
            className="desc"
            type="text"
            value={category}
            onChange={(e) => setcategory(e.target.value)}
            placeholder="Enter the Job Description"
          />
          <input
            className="num-candidates"
            type="number"
            value={numCandidates}
            onChange={(e) => setNumCandidates(e.target.value)}
            placeholder="Enter the number of candidates"
          />
        </div>
      </div>

      <div className="container">
        <form onSubmit={handleSubmit}>
          <div className="prompt-group">
            <input
              type="text"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Enter your prompt"
            />
            <button type="submit">Enter</button>
          </div>
        </form>
      </div>

      {output.category && (
        <div className="container output">
          <div>
            <h2>Job Description:</h2>
            <p>{output.category}</p>
          </div>
          <div>
            <h2>Prompt:</h2>
            <p>{output.prompt}</p>
          </div>
          <div>
            <h2>Number of Candidates:</h2>
            <p>{output.numCandidates}</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default PromptInput;
