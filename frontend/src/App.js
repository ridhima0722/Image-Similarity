
import React, { useState } from 'react';
import axios from 'axios';
import './App.css'

const App = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [similarImages, setSimilarImages] = useState([]);
  // console.log("this is selectedfile",selectedFile)
  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };
  

  const handleUpload = () => {
    const formData = new FormData();
    formData.append('file', selectedFile);
    setIsLoading(true);
    axios.post('http://localhost:5000/query_img/', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
      .then(response => {
        setSimilarImages(response.data.results);
        setIsLoading(false);
      })
      .catch(error => {
        console.error('Error uploading/querying image:', error);
      });
  };

  return (
    <div className='main'>
      <h1  className='primary'>Image Similarity </h1>
      <div className='image'>
      {selectedFile && <img src={URL.createObjectURL(selectedFile)} alt="Selected Image" />}
      </div>
      
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleUpload} >Upload</button>

      {similarImages.length > 0 && (
        <div >
        { isLoading ? <p>Loading...</p> : null}
          <h2 className='secondary'>Top Similar Images:</h2>
          {similarImages.map((result, index) => (
            <div key={index} className='second-main'>
              <p>Similarity: {result.similarity.toFixed(2)}</p>
              {/* <p>Similarity: {result.similarity.toFixed(2)}%</p> */}
              {/* <p>Image Path: {result.imgPath}</p> */}
              {/* <img src={`http://localhost:5000/${result.imgPath}`} alt={`Similar Image ${index + 1}`} />
               */}
               {/* <img src={`http://localhost:5000/images/${result.imgPath.split('\\').pop()}`} alt={`Similar Image ${index + 1}`} /> */}
               {/* <img src={`http://localhost:5000/images/${result.imgPath.split()[1]}`} alt={`Similar Image ${index + 1}`} /> */}
               {/* <img src={`http://localhost:5000${result.imgPath.split('/').pop()}`} alt={`Similar Image ${index + 1}`} /> */}
               <img src={result.imgPath} alt={`Similar Image ${index + 1}`} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
export default App;




