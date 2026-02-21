document.querySelector('.contact-form').addEventListener('submit', function(e) {
  e.preventDefault();

  const name = document.getElementById('name').value;
  const email = document.getElementById('email').value;
  const message = document.getElementById('message').value;
  const submitButton = this.querySelector('button');

  submitButton.textContent = 'Sending...';
  submitButton.disabled = true;

  fetch('/send', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ name, email, message }),
  })
  .then(response => response.json()) // Always expect a JSON response
  .then(data => {
    if (data.success) {
      // Success case
      alert('Thank you for reaching out');
      this.reset(); // Clear the form
    } else {
      // Server-side error case
      // We display the specific message from the server.
      // Only in this case do we also reveal the contact email.
      alert(data.message + '\n\nPlease contact us directly at greenjoy@globaltectonics.net');
    }
  })
  .catch(error => {
    // This catches network errors or if the server is completely down.
    console.error('Fetch Error:', error);
    alert("Couldn't connect to server. Check your internet.");
  })
  .finally(() => {
    submitButton.textContent = 'Connect';
    submitButton.disabled = false;
  });
});
