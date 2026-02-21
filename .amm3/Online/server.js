const express = require('express');
const nodemailer = require('nodemailer');
const cors = require('cors');
const path = require('path');
require('dotenv').config();

const app = express();
const port = 3000;

// Middleware
app.use(cors());
app.use(express.json());

// Serve static files from the 'GlobalTectonics.net' subdirectory
app.use(express.static(path.join(__dirname, 'GlobalTectonics.net')));

// POST route for sending email
app.post('/send', async (req, res) => {
  const { name, email, message } = req.body;

  const transporter = nodemailer.createTransport({
    host: 'smtp.office365.com',
    port: 587,
    secure: false,
    auth: {
      user: process.env.EMAIL_USER,
      pass: process.env.EMAIL_PASS,
    },
  });

  try {
    await transporter.sendMail({
      from: "GlobalTectonics.net" <${process.env.EMAIL_USER}>,
      to: process.env.EMAIL_USER,
      replyTo: email,
      subject: name,
      text: `Email: ${email}
Message: ${message}`,
    });
    
    console.log('Email sent successfully');
    res.status(200).json({ success: true, message: 'Email sent successfully' });

  } catch (error) {
    console.error('Error sending email:', error);

    if (error.code === 'EAUTH') {
      res.status(500).json({ success: false, message: 'A server authentication error occurred.' });
    } else if (error.code === 'ECONNRESET' || error.code === 'ETIMEDOUT') {
      res.status(500).json({ success: false, message: 'The email server is not responding. Please try again later.' });
    } else {
      res.status(500).json({ success: false, message: 'An unexpected error occurred.' });
    }
  }
});

// API endpoint to provide the current server time
app.get('/api/time', (req, res) => {
  res.json({ serverTime: Date.now() });
});

app.listen(port, () => {
  console.log(`Server listening at http://localhost:${port}`);
});
