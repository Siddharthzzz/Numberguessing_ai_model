// --- Canvas Setup ---
const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');

const clearBtn = document.getElementById('clearBtn');
const predictBtn = document.getElementById('predictBtn');
const resultBox = document.getElementById('resultBox');
const predictionDisplay = document.getElementById('prediction');
const confidenceDisplay = document.getElementById('confidence');
const errorMessage = document.getElementById('errorMessage');

// --- IMPORTANT: API ENDPOINT ---
// This is the address of your Python Flask server.
// For local testing, it must be http://127.0.0.1:5000/predict
const apiEndpoint = 'http://127.0.0.1:5000/predict';

// --- INITIALIZE CANVAS ---
let isDrawing = false;
let hasDrawn = false;
let lastX = 0;
let lastY = 0;

function initializeCanvas() {
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 20;
    ctx.lineCap = 'round';
    ctx.strokeStyle = '#FFFFFF';
    hasDrawn = false;
}

// --- DRAWING FUNCTIONS ---
function draw(e) {
    if (!isDrawing) return;

    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX || e.touches[0].clientX) - rect.left;
    const y = (e.clientY || e.touches[0].clientY) - rect.top;

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();
    [lastX, lastY] = [x, y];
    hasDrawn = true;
}

function startDrawing(e) {
    // Only draw with the left mouse button
    if (e.button && e.button !== 0) return;

    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX || e.touches[0].clientX) - rect.left;
    const y = (e.clientY || e.touches[0].clientY) - rect.top;
    [lastX, lastY] = [x, y];
}

const stopDrawing = () => { isDrawing = false; };

// --- EVENT LISTENERS ---
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);
canvas.addEventListener('touchstart', startDrawing, { passive: false });
canvas.addEventListener('touchmove', draw, { passive: false });
canvas.addEventListener('touchend', stopDrawing);

// --- CONTROL BUTTONS ---
clearBtn.addEventListener('click', () => {
    initializeCanvas();
    predictionDisplay.textContent = '-';
    confidenceDisplay.textContent = '-';
    resultBox.classList.remove('success', 'fail');
    hideError();
});

predictBtn.addEventListener('click', () => {
    if (!hasDrawn) {
        showError('Please draw a digit first before guessing!');
        return;
    }
    const imageDataURL = canvas.toDataURL('image/png');
    sendToModelBackend(imageDataURL);
});

// --- API INTERACTION ---
async function sendToModelBackend(imageData, retries = 3, delay = 1000) {
    showLoading();
    hideError();

    for (let i = 0; i < retries; i++) {
        try {
            const response = await fetch(apiEndpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageData })
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: 'Server returned an invalid response.' }));
                throw new Error(`Server error: ${errorData.error || response.statusText}`);
            }

            const result = await response.json();
            displayPrediction(result.prediction, result.confidence);
            return; // Success, exit the loop
        } catch (error) {
            console.error(`Attempt ${i + 1} failed:`, error);
            if (i === retries - 1) {
                // Last attempt failed
                displayFailState(`Failed to connect to ML server. Is it running?`);
            } else {
                // Wait before retrying
                await new Promise(resolve => setTimeout(resolve, delay * Math.pow(2, i)));
            }
        }
    }
}

// --- UI UPDATE FUNCTIONS ---
function showLoading() {
    predictionDisplay.textContent = '...';
    confidenceDisplay.textContent = '...';
    resultBox.classList.remove('success', 'fail');
}

function displayPrediction(prediction, confidence) {
    predictionDisplay.textContent = prediction;
    confidenceDisplay.textContent = `${(confidence * 100).toFixed(2)}%`;
    resultBox.classList.add('success');
    resultBox.classList.remove('fail');
}

function displayFailState(message) {
    predictionDisplay.textContent = 'FAIL';
    confidenceDisplay.textContent = 'Check Console';
    resultBox.classList.add('fail');
    resultBox.classList.remove('success');
    showError(message);
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
}

function hideError() {
    errorMessage.style.display = 'none';
}

// --- INITIALIZE ---
window.onload = initializeCanvas;

