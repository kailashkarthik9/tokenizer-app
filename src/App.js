import React, { useState } from 'react';
import './App.css';

function App() {
  const [modelName, setModelName] = useState('bert-base-uncased');
  const [text, setText] = useState('');
  const [tokens, setTokens] = useState([]);
  const [tokenCount, setTokenCount] = useState(0);
  const [isTokenized, setIsTokenized] = useState(false);
  const [offsets, setOffsets] = useState([]);
  const [hoveredToken, setHoveredToken] = useState(null);

  const handleTokenize = async () => {
    const response = await fetch('/api/tokenize', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 'model-name': modelName, text }),
    });

    const result = await response.json();
    setTokens(result.tokens || []); // Ensure tokens is an array
    setOffsets(result.offsets || []); // Ensure offsets is an array
    setTokenCount(result.token_count || 0); // Ensure token_count is a number
    setIsTokenized(true); // Mark that tokenization is complete
  };

  const handleTextChange = (e) => {
    setText(e.target.value);
    setIsTokenized(false); // Mark that new input is being entered
    setTokens([]);
    setOffsets([]);
    setTokenCount(0);
  };

  const handleModelNameChange = (e) => {
    setModelName(e.target.value);
    setIsTokenized(false); // Mark that new input is being entered
    setTokens([]);
    setOffsets([]);
    setTokenCount(0);
  };

  const handleMouseEnter = (index) => {
    setHoveredToken(index);
  };

  const handleMouseLeave = () => {
    setHoveredToken(null);
  };

  const renderTokens = () => {
    const elements = [];
    let currentIndex = 0;

    offsets.forEach((offset, index) => {
      const [start, end] = offset;
      // Add any text between the last token and the current token
      if (start > currentIndex) {
        elements.push(
            <span key={`text-${currentIndex}`}>
            {text.slice(currentIndex, start).split('\n').map((segment, i) => (
                <React.Fragment key={`segment-${currentIndex}-${i}`}>
                  {segment}
                  {i < text.slice(currentIndex, start).split('\n').length - 1 && <br />}
                </React.Fragment>
            ))}
          </span>
        );
      }
      // Tokenized spans
      elements.push(
          <span
              key={`token-${index}`}
              className={`token color-${index % 10} ${hoveredToken !== null && hoveredToken !== index ? 'hidden' : ''}`}
              onMouseEnter={() => handleMouseEnter(index)}
              onMouseLeave={handleMouseLeave}
          >
          {text.slice(start, end)}
            {text[start] === '\n' && <br />}
        </span>
      );
      currentIndex = end;
    });

    // Add any remaining text after the last token
    if (currentIndex < text.length) {
      elements.push(
          <span key={`text-${currentIndex}`}>
          {text.slice(currentIndex).split('\n').map((segment, i) => (
              <React.Fragment key={`segment-${currentIndex}-${i}`}>
                {segment}
                {i < text.slice(currentIndex).split('\n').length - 1 && <br />}
              </React.Fragment>
          ))}
        </span>
      );
    }

    return elements;
  };

  return (
      <div className="App">
        <header className="App-header">
          <h1>Tokenizer</h1>
          <input
              type="text"
              placeholder="Enter model name"
              value={modelName}
              onChange={handleModelNameChange}
          />
          <textarea
              placeholder="Enter text here"
              value={text}
              onChange={handleTextChange}
          />
          <button onClick={handleTokenize}>Tokenize</button>
          <div>Token count: {tokenCount}</div>
          <div className="tokens">
            {isTokenized ? renderTokens() : <p>Enter text and click "Tokenize" to see tokenized output.</p>}
          </div>
        </header>
      </div>
  );
}

export default App;