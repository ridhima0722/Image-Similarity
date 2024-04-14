import React from 'react';

const ImageResults= ({ results }) => {
  return (
    <div>
      <h2>Top 5 Similar Images:</h2>
      <ul>
        {results.map((result, index) => (
          <li key={index}>
            <img src={`http://localhost:5000/get_image/${encodeURIComponent(result.imgPath)}`} alt={`Result ${index + 1}`} />
            <p>Similarity: {result.similarity}</p>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default ImageResults;