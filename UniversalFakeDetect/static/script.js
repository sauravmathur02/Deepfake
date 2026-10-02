const uploadArea    = document.getElementById('uploadArea');
const fileInput     = document.getElementById('fileInput');
const uploadContent = document.querySelector('.upload-content');
const previewArea   = document.getElementById('previewArea');
const imagePreview  = document.getElementById('imagePreview');
const videoPreview  = document.getElementById('videoPreview');
const changeImageBtn= document.getElementById('changeImageBtn');
const analyzeBtn    = document.getElementById('analyzeBtn');
const resultsArea   = document.getElementById('resultsArea');
const loader        = document.getElementById('loader');
const loaderText    = document.getElementById('loaderText');
const predictionCard= document.getElementById('predictionCard');
const scoreValue    = document.getElementById('scoreValue');
const verdict       = document.getElementById('verdict');
const scoreCircle   = document.getElementById('scoreCircle');
const scoreMarker   = document.getElementById('scoreMarker');
const detailsText   = document.getElementById('detailsText');
const clipScoreEl   = document.getElementById('clipScore');
const fftScoreEl    = document.getElementById('fftScore');
const edgeScoreEl   = document.getElementById('edgeScore');

let currentFile = null;

// ── Drag & Drop ────────────────────────────────────────────────────────────────
['dragenter','dragover','dragleave','drop'].forEach(e =>
    uploadArea.addEventListener(e, ev => { ev.preventDefault(); ev.stopPropagation(); })
);
['dragenter','dragover'].forEach(e =>
    uploadArea.addEventListener(e, () => { if (!currentFile) uploadArea.classList.add('dragover'); })
);
['dragleave','drop'].forEach(e =>
    uploadArea.addEventListener(e, () => uploadArea.classList.remove('dragover'))
);
uploadArea.addEventListener('drop', e => {
    if (!currentFile) handleFiles(e.dataTransfer.files);
});
fileInput.addEventListener('change', function() { handleFiles(this.files); });

function handleFiles(files) {
    if (!files.length) return;
    const file = files[0];
    if (file.type.startsWith('image/') || file.type.startsWith('video/')) {
        currentFile = file;
        displayPreview(file);
    } else {
        alert('Please upload an image or video file.');
    }
}

function displayPreview(file) {
    const url = URL.createObjectURL(file);
    if (file.type.startsWith('video/')) {
        imagePreview.classList.add('hidden');
        videoPreview.classList.remove('hidden');
        videoPreview.src = url;
    } else {
        videoPreview.classList.add('hidden');
        imagePreview.classList.remove('hidden');
        imagePreview.src = url;
    }
    uploadContent.classList.add('hidden');
    previewArea.classList.remove('hidden');
    analyzeBtn.disabled = false;
    analyzeBtn.classList.remove('disabled');
    resultsArea.classList.add('hidden');
    predictionCard.classList.add('hidden');
}

changeImageBtn.addEventListener('click', e => {
    e.stopPropagation();
    currentFile = null;
    fileInput.value = '';
    if (imagePreview.src) URL.revokeObjectURL(imagePreview.src);
    if (videoPreview.src) URL.revokeObjectURL(videoPreview.src);
    imagePreview.src = videoPreview.src = '';
    previewArea.classList.add('hidden');
    uploadContent.classList.remove('hidden');
    analyzeBtn.disabled = true;
    analyzeBtn.classList.add('disabled');
    resultsArea.classList.add('hidden');
});

uploadArea.addEventListener('click', () => { if (!currentFile) fileInput.click(); });

// ── Analyze ────────────────────────────────────────────────────────────────────
analyzeBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    analyzeBtn.disabled = true;
    analyzeBtn.classList.add('disabled');
    resultsArea.classList.remove('hidden');
    loader.classList.remove('hidden');
    predictionCard.classList.add('hidden');

    const isVideo = currentFile.type.startsWith('video/');
    loaderText.textContent = isVideo
        ? 'Extracting frames & running 3-layer ensemble...'
        : 'Running 3-layer ensemble analysis...';

    const formData = new FormData();
    formData.append('file', currentFile);

    try {
        const res  = await fetch('/predict', { method:'POST', body:formData });
        const data = await res.json();
        if (data.success) {
            displayResults(data);
        } else {
            const msg = data.error || 'Unknown server error. Check that the file is a valid image or video.';
            alert('Analysis failed: ' + msg);
            resetState();
        }
    } catch(err) {
        alert('Could not reach server. Is it running?');
        resetState();
    }
});

function resetState() {
    analyzeBtn.disabled = false;
    analyzeBtn.classList.remove('disabled');
    loader.classList.add('hidden');
    resultsArea.classList.add('hidden');
}

// ── Display results ────────────────────────────────────────────────────────────
function displayResults(data) {
    loader.classList.add('hidden');
    predictionCard.classList.remove('hidden');

    const isFake = data.prediction === 'Fake';

    // Main verdict
    scoreValue.textContent = data.confidence;
    verdict.className = 'verdict ' + (isFake ? 'fake' : 'real');
    verdict.textContent = data.prediction.toUpperCase();

    // Score circle colour
    const col = isFake ? 'var(--fake-color)' : 'var(--real-color)';
    const glow = isFake ? 'rgba(239,68,68,.4)' : 'rgba(16,185,129,.4)';
    scoreCircle.style.borderColor = col;
    scoreCircle.style.color       = col;
    scoreCircle.style.boxShadow   = `0 0 30px ${glow}`;

    // Animate marker
    setTimeout(() => { scoreMarker.style.left = `${data.score * 100}%`; }, 100);

    // Breakdown cards
    renderDetector('card-clip', clipScoreEl, data.clip_score, 'CLIP');
    renderDetector('card-fft',  fftScoreEl,  data.fft_score,  'FFT');
    renderDetector('card-edge', edgeScoreEl, data.edge_score, 'Edge');

    detailsText.textContent =
        `Ensemble score: ${data.score.toFixed(6)}` +
        (data.frames_analyzed > 1 ? ` · ${data.frames_analyzed} frames averaged` : '');

    analyzeBtn.disabled = false;
    analyzeBtn.classList.remove('disabled');
}

function renderDetector(cardId, scoreEl, score, label) {
    const card     = document.getElementById(cardId);
    const pct      = Math.round(score * 100);
    const isFakeSig = score > 0.45;

    scoreEl.textContent = pct + '%';
    scoreEl.style.color  = isFakeSig ? 'var(--fake-color)' : 'var(--real-color)';
    card.className = 'detector-card ' + (isFakeSig ? 'signal-fake' : 'signal-real');
}
