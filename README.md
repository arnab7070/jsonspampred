# Spam Predictor API
A minimal Flask web application hosted on vercel
## Getting Started

These instructions will help you get your API up and running. Make sure that you make the model in json format so that it cannot
exceed more than 250MB serverless function.

### Prerequisites

- Node.js installed
- npm (Node Package Manager) installed
- axios library installed (you can install it using `npm install axios`)

### Example
```js
const axios = require('axios');

async function makePrediction(commentText) {
  try {
    const response = await axios.post('https://jsonspampred.vercel.app/predict', {'comment': commentText});
    const responseData = response.data;
    console.log('Prediction:', responseData);
  } catch (error) {
    console.error('Error making prediction:', error.message);
  }
}

// Example usage
const commentText = 'This is a sample comment.';
makePrediction(commentText);
```
### API Documentation
----------------------
Document the available endpoints, request parameters, and response format

#### Endpoint: 
- POST /predict
#### Request Body:
- comment (string): The text comment for prediction.
#### Response:
- result (bool): The prediction result.
  ```json
  {
    "result": true
  }
  ```