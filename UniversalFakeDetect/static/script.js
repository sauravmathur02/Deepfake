const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadContent = document.querySelector('.upload-content');
const previewArea = document.getElementById('previewArea');
const imagePreview = document.getElementById('imagePreview');
const videoPreview = document.getElementById('videoPreview');
const changeImageBtn = document.getElementById('changeImageBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultsArea = document.getElementById('resultsArea');
const loader = document.getElementById('loader');
const predictionCard = document.getElementById('predictionCard');
const scoreValue = document.getElementById('scoreValue');
const verdict = document.getElementById('verdict');
const scoreCircle = document.getElementById('scoreCircle');
const scoreMarker = document.getElementById('scoreMarker');
const detailsText = document.getElementById('detailsText');

let currentFile = null;

// Drag and Drop Events
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    uploadArea.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    uploadArea.addEventListener(eventName, () => {
        if (!currentFile) uploadArea.classList.add('dragover');
    }, false);
});

['dragleave', 'drop'].forEach(eventName => {
    uploadArea.addEventListener(eventName, () => {
        uploadArea.classList.remove('dragover');
    }, false);
});

uploadArea.addEventListener('drop', handleDrop, false);

function handleDrop(e) {
    if (currentFile) return;
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
}

fileInput.addEventListener('change', function() {
    handleFiles(this.files);
});

function handleFiles(files) {
    if (files.length > 0) {
        const file = files[0];
        if (file.type.startsWith('image/') || file.type.startsWith('video/')) {
            currentFile = file;
            displayPreview(file);
        } else {
            alert('Please upload an image or video file.');
        }
    }
}

function displayPreview(file) {
    const objectUrl = URL.createObjectURL(file);
    
    if (file.type.startsWith('video/')) {
        imagePreview.classList.add('hidden');
        videoPreview.classList.remove('hidden');
        videoPreview.src = objectUrl;
    } else {
        videoPreview.classList.add('hidden');
        imagePreview.classList.remove('hidden');
        imagePreview.src = objectUrl;
    }
    
    uploadContent.classList.add('hidden');
    previewArea.classList.remove('hidden');
    analyzeBtn.classList.remove('disabled');
    analyzeBtn.disabled = false;
    
    // Hide previous results
    resultsArea.classList.add('hidden');
    predictionCard.classList.add('hidden');
}

changeImageBtn.addEventListener('click', (e) => {
    e.stopPropagation(); // Prevent triggering uploadArea click
    currentFile = null;
    fileInput.value = '';
    
    if (imagePreview.src) URL.revokeObjectURL(imagePreview.src);
    if (videoPreview.src) URL.revokeObjectURL(videoPreview.src);
    
    imagePreview.src = '';
    videoPreview.src = '';
    
    previewArea.classList.add('hidden');
    uploadContent.classList.remove('hidden');
    
    analyzeBtn.classList.add('disabled');
    analyzeBtn.disabled = true;
    
    resultsArea.classList.add('hidden');
});

uploadArea.addEventListener('click', () => {
    if (!currentFile) {
        fileInput.click();
    }
});

// Analyze Button Click
analyzeBtn.addEventListener('click', async () => {
    if (!currentFile) return;
    
    // UI Updates
    analyzeBtn.disabled = true;
    analyzeBtn.classList.add('disabled');
    resultsArea.classList.remove('hidden');
    loader.classList.remove('hidden');
    predictionCard.classList.add('hidden');
    
    // Prepare Data
    const formData = new FormData();
    formData.append('file', currentFile);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data);
        } else {
            alert('Error analyzing media: ' + data.error);
            resetAnalyzeState();
        }
    } catch (error) {
        alert('Failed to connect to the server. Is it running?');
        resetAnalyzeState();
    }
});

function resetAnalyzeState() {
    analyzeBtn.disabled = false;
    analyzeBtn.classList.remove('disabled');
    loader.classList.add('hidden');
    resultsArea.classList.add('hidden');
}

function displayResults(data) {
    loader.classList.add('hidden');
    predictionCard.classList.remove('hidden');
    
    // Update Score Circle and Verdict
    scoreValue.textContent = data.confidence;
    verdict.textContent = data.prediction.toUpperCase();
    
    // Reset classes
    verdict.className = 'verdict';
    scoreCircle.style.borderColor = '';
    
    // Calculate position for marker (0 to 100%)
    // Score is 0 (Real) to 1 (Fake)
    const markerPosition = data.score * 100;
    
    // Trigger animation for the marker
    setTimeout(() => {
        scoreMarker.style.left = `${markerPosition}%`;
    }, 100);
    
    if (data.prediction === 'Fake') {
        verdict.classList.add('fake');
        scoreCircle.style.borderColor = 'var(--fake-color)';
        scoreCircle.style.color = 'var(--fake-color)';
        scoreCircle.style.boxShadow = '0 0 30px rgba(239, 68, 68, 0.4)';
    } else {
        verdict.classList.add('real');
        scoreCircle.style.borderColor = 'var(--real-color)';
        scoreCircle.style.color = 'var(--real-color)';
        scoreCircle.style.boxShadow = '0 0 30px rgba(16, 185, 129, 0.4)';
    }
    
    detailsText.textContent = `Raw model score: ${data.score.toFixed(6)}`;
    
    // Re-enable analyze button in case they want to run it again
    analyzeBtn.disabled = false;
    analyzeBtn.classList.remove('disabled');
}
