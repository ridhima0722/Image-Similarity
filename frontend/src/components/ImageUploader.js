// import React,{useState} from "react";
// import axios from "axios";

// class Main extends React.Component {
//   state = {
//     // Initially, no file is selected
//     selectedFile: null,
//   };

// // On file select (from the pop up)
//   onFileChange = (event) => {
//     // Update the state
//     this.setState({
//         selectedFile: event.target.files[0],
//     });
//   };
//   handleUploadImage(ev) {
//     ev.preventDefault();

//     const data = new FormData();
//     data.append('file', this.uploadInput.files[0]);
//     data.append('filename', this.fileName.value);

//      // Details of the uploaded file
//      console.log(this.state.selectedFile);
 
//      // Request made to the backend api
//      // Send formData object
//      axios.post("api/uploadfile", formData);
//   };
//   render() {
//     return (
//       <form onSubmit={this.handleUploadImage}>
//         <div>
//           <input ref={(ref) => { this.uploadInput = ref; }} type="file" />
//         </div>
//         {/* <div>
//           <input ref={(ref) => { this.fileName = ref; }} type="text" placeholder="Enter the desired name of file" />
//         </div> */}
//         <br />
//         <div>
//           <button>Upload</button>
//         </div>
//         <img src={this.state.imageURL} alt="img" />
//       </form>
//     );
//   }
// } 
// export default Main;



import React, { useState } from 'react';
import axios from 'axios';

const ImageUploader= ({ onUpload }) => {
  const [selectedFile, setSelectedFile] = useState(null);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post('http://localhost:5000/query_img/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      onUpload(response.data.results);
    } catch (error) {
      console.error('Error uploading image:', error);
    }
  };

  return (
    <div>
      <input type="file" accept=".jpg, .jpeg, .png" onChange={handleFileChange} />
      <button onClick={handleUpload}>insert</button>
    </div>
  );
};

export default ImageUploader;
