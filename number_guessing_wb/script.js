// --- Configuration and Elements ---
const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');

const clearBtn = document.getElementById('clearBtn');
const predictBtn = document.getElementById('predictBtn');
const predictionDisplay = document.getElementById('prediction');
const confidenceDisplay = document.getElementById('confidence');
const messageArea = document.getElementById('messageArea');

// --- Drawing State Setup ---
let isDrawing = false;
let lastX = 0;
let lastY = 0;

// Set up the drawing style: Thick white line for the 'number' on black canvas
ctx.lineWidth = 20;       
ctx.lineCap = 'round';    
ctx.strokeStyle = '#ffffff'; 

// --- Utility Functions ---

/**
 * Displays a temporary error message in the message area.
 * @param {string} text - The message to display.
 */
function showMessage(text) {
    // Clear existing messages
    messageArea.innerHTML = ''; 

    const message = document.createElement('div');
    message.className = 'message-error';
    message.innerHTML = `<i class="fas fa-exclamation-triangle mr-2"></i> ${text}`;
    messageArea.appendChild(message);

    // Remove the message after 7 seconds
    setTimeout(() => message.remove(), 7000); 
}

// --- Drawing Functionality ---

/**
 * Draws a line segment between the last recorded position and the current position.
 * @param {Event} e - Mouse or Touch event.
 */
function draw(e) {
    if (!isDrawing) return;

    // Determine if it's a mouse or touch event
    const clientX = e.clientX || e.touches[0].clientX;
    const clientY = e.clientY || e.touches[0].clientY;

    // Get position relative to the canvas
    const rect = canvas.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;

    ctx.beginPath();
    ctx.moveTo(lastX, lastY); 
    ctx.lineTo(x, y);         
    ctx.stroke();

    // Update last position for the next segment
    [lastX, lastY] = [x, y];
}

/**
 * Starts the drawing process (mousedown or touchstart).
 * @param {Event} e - Mouse or Touch event.
 */
function startDrawing(e) {
    e.preventDefault(); 
    // Only proceed if it's a touch event OR a left mouse button click (button === 0)
    if (e.touches || (e.button === 0)) { 
        isDrawing = true;

        const clientX = e.clientX || e.touches[0].clientX;
        const clientY = e.clientY || e.touches[0].clientY;

        const rect = canvas.getBoundingClientRect();
        const x = clientX - rect.left;
        const y = clientY - rect.top;

        [lastX, lastY] = [x, y];
        draw(e); 
    }
}

/**
 * Stops the drawing process.
 */
function stopDrawing() {
    isDrawing = false;
}

// Event Listeners for Drawing (Mouse & Touch)
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

canvas.addEventListener('touchstart', startDrawing);
canvas.addEventListener('touchmove', draw);
canvas.addEventListener('touchend', stopDrawing);


// --- Control Event Listeners ---

// Clear the canvas and results
clearBtn.addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Re-fill the canvas with black background 
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height); 
    
    // Reset the result display and messages
    predictionDisplay.textContent = '-';
    confidenceDisplay.textContent = '-';
    messageArea.innerHTML = '';
});

// Trigger prediction
predictBtn.addEventListener('click', () => {
    // Check if drawing area is actually used (simple check: if any pixel is not black)
    const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    let isBlank = true;
    // Check every 4th pixel (R component)
    for (let i = 0; i < data.length; i += 4) {
        if (data[i] !== 0 || data[i+1] !== 0 || data[i+2] !== 0) { 
            isBlank = false;
            break;
        }
    }

    if (isBlank) {
        showMessage('Please draw a number before clicking Guess!');
        return;
    }

    const imageDataURL = canvas.toDataURL('image/png'); 
    sendToModelBackend(imageDataURL);
});

// --- Backend/API Interaction (Placeholder Function with Exponential Backoff) ---
async function sendToModelBackend(imageData) {
    // ⚠️ IMPORTANT: YOU MUST REPLACE THIS WITH YOUR DEPLOYED MODEL'S PUBLIC URL
    const apiEndpoint = 'YOUR_BACKEND_API_ENDPOINT/predict'; 

    // Visual feedback while waiting
    predictionDisplay.textContent = '...';
    confidenceDisplay.textContent = '...';
    messageArea.innerHTML = '';

    const maxRetries = 3;
    let attempt = 0;

    while (attempt < maxRetries) {
        attempt++;
        try {
            const response = await fetch(apiEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ image: imageData })
            });

            if (!response.ok) {
                // Throw an error to trigger the catch block and retry logic
                throw new Error(`HTTP error! Status: ${response.status}`);
            }

            const result = await response.json();
            
            // Assuming result is { "prediction": 7, "confidence": 0.985 }
            const predictedNumber = result.prediction || '?';
            const confidenceScore = result.confidence ? (result.confidence * 100).toFixed(2) : '??';

            predictionDisplay.textContent = predictedNumber;
            confidenceDisplay.textContent = `${confidenceScore}%`;
            return; // Success, exit the loop

        } catch (error) {
            console.error(`Attempt ${attempt} failed:`, error);
            
            if (attempt < maxRetries) {
                const delay = Math.pow(2, attempt) * 1000; // Exponential backoff (2s, 4s, 8s)
                predictionDisplay.textContent = `...Retrying (${attempt}/${maxRetries})...`;
                await new Promise(resolve => setTimeout(resolve, delay));
            } else {
                // Final failure
                predictionDisplay.textContent = 'FAIL';
                confidenceDisplay.textContent = 'Check Console';
                showMessage(`Failed to connect to ML server after ${maxRetries} tries. Update API endpoint.`);
            }
        }
    }
}

// Initial setup: ensure the canvas starts black
window.onload = function() {
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
};