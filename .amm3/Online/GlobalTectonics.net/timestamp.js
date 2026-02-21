// This function will be called once we get the time from the server.
function startClocks(serverTime) {
  // Create a Date object based on the server's timestamp.
  // This is our "master clock".
  const masterClock = new Date(serverTime);

  const timeOptions = {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false
  };

  function updateClocks() {
    // Instead of creating a new Date(), we just add one second
    // to our master clock.
    masterClock.setSeconds(masterClock.getSeconds() + 1);

    // Now, display the time from our master clock in different timezones.
    document.getElementById('time-ireland').textContent = masterClock.toLocaleTimeString('en-GB', {
      ...timeOptions,
      timeZone: 'Europe/Dublin'
    });

    document.getElementById('time-saigon').textContent = masterClock.toLocaleTimeString('en-GB', {
      ...timeOptions,
      timeZone: 'Asia/Ho_Chi_Minh'
    });

    document.getElementById('time-melbourne').textContent = masterClock.toLocaleTimeString('en-GB', {
      ...timeOptions,
      timeZone: 'Australia/Melbourne'
    });

    document.getElementById('time-pennsylvania').textContent = masterClock.toLocaleTimeString('en-GB', {
      ...timeOptions,
      timeZone: 'America/New_York'
    });

    document.getElementById('time-prague').textContent = masterClock.toLocaleTimeString('en-GB', {
      ...timeOptions,
      timeZone: 'Europe/Prague'
    });

    document.getElementById('time-singapore').textContent = masterClock.toLocaleTimeString('en-GB', {
      ...timeOptions,
      timeZone: 'Asia/Singapore'
    });

    document.getElementById('time-wairoa').textContent = masterClock.toLocaleTimeString('en-GB', {
      ...timeOptions,
      timeZone: 'Pacific/Auckland'
    });
  }

  // Run the update function immediately to show the time,
  // then start the interval to update it every second.
  updateClocks();
  setInterval(updateClocks, 1000);
}

// When the page loads, fetch the master time from the server.
fetch('/api/time')
  .then(response => response.json())
  .then(data => {
    // Once we have the server's time, start the clocks.
    startClocks(data.serverTime);
  });
