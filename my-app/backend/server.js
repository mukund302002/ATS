// backend/server.js
const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');

const app = express();
const PORT = 5000;

app.use(cors());
app.use(bodyParser.json());

app.post('/api/prompt', (req, res) => {
    const { jobDescription, prompt, numCandidates } = req.body;
    res.json({ jobDescription, prompt, numCandidates });
  });

app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
