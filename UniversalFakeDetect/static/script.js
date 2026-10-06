const uploadArea    = document.getElementById('uploadArea');
const fileInput     = document.getElementById('fileInput');
const uploadContent = document.querySelector('.upload-content');
const previewArea   = document.getElementById('previewArea');
const imagePreview  = document.getElementById('imagePreview');
const videoPreview  = document.getElementById('videoPreview');
const changeImageBtn= document.getElementById('changeImageBtn');

const bulkPreviewArea = document.getElementById('bulkPreviewArea');
const bulkCountText   = document.getElementById('bulkCountText');
const bulkChangeBtn   = document.getElementById('bulkChangeBtn');

const analyzeBtn      = document.getElementById('analyzeBtn');

const resultsArea     = document.getElementById('resultsArea');
const analysisProcess = document.getElementById('analysisProcess');
const predictionCard  = document.getElementById('predictionCard');
const scoreValue      = document.getElementById('scoreValue');
const verdict         = document.getElementById('verdict');
const scoreCircle     = document.getElementById('scoreCircle');
const scoreMarker     = document.getElementById('scoreMarker');
const detailsText     = document.getElementById('detailsText');
const commforScoreEl  = document.getElementById('commforScore');
const bfreeScoreEl    = document.getElementById('bfreeScore');
const provScoreEl     = document.getElementById('provScore');
const gendScoreEl     = document.getElementById('gendScore');


const bulkResultsArea = document.getElementById('bulkResultsArea');
const bulkProgressContainer = document.getElementById('bulkProgressContainer');
const bulkProgressText= document.getElementById('bulkProgressText');
const bulkProgressBar = document.getElementById('bulkProgressBar');
const bulkGridContainer = document.getElementById('bulkGridContainer');
const resultsGrid = document.getElementById('results-grid');

let currentFiles = [];
let feedbackData = {};
let results = {};
const emptyState = document.getElementById('emptyState');
const statsPanel = document.getElementById('statsPanel');

function updateTally() {
    const vals = Object.values(results);
    document.getElementById('tallyReal').textContent = vals.filter(r => r.prediction === 'Real').length;
    document.getElementById('tallyFake').textContent = vals.filter(r => r.prediction === 'Fake').length;
}

// ── Drag & Drop ────────────────────────────────────────────────────────────────
['dragenter','dragover','dragleave','drop'].forEach(e =>
    uploadArea.addEventListener(e, ev => { ev.preventDefault(); ev.stopPropagation(); })
);
['dragenter','dragover'].forEach(e =>
    uploadArea.addEventListener(e, () => { if (!currentFiles.length) uploadArea.classList.add('dragover'); })
);
['dragleave','drop'].forEach(e =>
    uploadArea.addEventListener(e, () => uploadArea.classList.remove('dragover'))
);
uploadArea.addEventListener('drop', e => {
    if (!currentFiles.length) handleFiles(e.dataTransfer.files);
});
fileInput.addEventListener('change', function() { handleFiles(this.files); });
uploadArea.addEventListener('click', () => { if (!currentFiles.length) fileInput.click(); });

function handleFiles(files) {
    if (!files.length) return;
    
    // Filter out non-media
    const valid = Array.from(files).filter(f => f.type.startsWith('image/') || f.type.startsWith('video/'));
    if (!valid.length) {
        alert('Please upload image or video files.');
        return;
    }

    currentFiles = valid;
    
    uploadContent.classList.add('hidden');
    analyzeBtn.disabled = false;
    analyzeBtn.classList.remove('disabled');
    resultsArea.classList.add('hidden');
    predictionCard.classList.add('hidden');
    bulkResultsArea.classList.add('hidden');

    if (currentFiles.length === 1) {
        displayPreview(currentFiles[0]);
    } else {
        displayBulkPreview();
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
    
    // Render Metadata
    const sizeStr = (file.size / (1024*1024)).toFixed(2);
    document.getElementById('fileMetadata').innerHTML = `<strong>File:</strong> ${file.name} &nbsp;|&nbsp; <strong>Type:</strong> ${file.type} &nbsp;|&nbsp; <strong>Size:</strong> ${sizeStr} MB`;
    
    previewArea.classList.remove('hidden');
    bulkPreviewArea.classList.add('hidden');
}

function displayBulkPreview() {
    previewArea.classList.add('hidden');
    bulkCountText.textContent = `${currentFiles.length} files selected`;
    bulkPreviewArea.classList.remove('hidden');
}

function clearFiles(e) {
    if(e) e.stopPropagation();
    currentFiles = [];
    feedbackData = {};
    results = {};
    updateTally();
    emptyState.classList.remove('hidden');
    statsPanel.classList.add('hidden');
    fileInput.value = '';
    if (imagePreview.src) URL.revokeObjectURL(imagePreview.src);
    if (videoPreview.src) URL.revokeObjectURL(videoPreview.src);
    imagePreview.src = videoPreview.src = '';
    
    previewArea.classList.add('hidden');
    bulkPreviewArea.classList.add('hidden');
    uploadContent.classList.remove('hidden');
    
    analyzeBtn.disabled = true;
    analyzeBtn.classList.add('disabled');
    resultsArea.classList.add('hidden');
    bulkResultsArea.classList.add('hidden');
    
    document.getElementById('generateReportBtn').style.display = 'none';
    document.getElementById('evaluationReport').classList.add('hidden');
}
changeImageBtn.addEventListener('click', clearFiles);
bulkChangeBtn.addEventListener('click', clearFiles);

// ── Analyze (Single) ──────────────────────────────────────────────────────────
function setStepState(stepId, state) {
    const step = document.getElementById(stepId);
    if (!step) return;
    step.className = `step ${state}`;
    const icon = step.querySelector('.step-icon i');
    if (state === 'active') {
        icon.className = 'fa-solid fa-circle-notch fa-spin';
    } else if (state === 'done') {
        icon.className = 'fa-solid fa-check-circle';
    } else {
        icon.className = 'fa-solid fa-circle';
    }
}

async function animateSteps() {
    for(let i=1; i<=5; i++) setStepState('step'+i, 'pending');
    
    setStepState('step1', 'active');
    await new Promise(r => setTimeout(r, 600));
    setStepState('step1', 'done');
    
    setStepState('step2', 'active');
    await new Promise(r => setTimeout(r, 800));
    setStepState('step2', 'done');
    
    setStepState('step3', 'active');
    await new Promise(r => setTimeout(r, 1200));
    setStepState('step3', 'done');
    
    setStepState('step4', 'active');
    await new Promise(r => setTimeout(r, 700));
    setStepState('step4', 'done');
    
    setStepState('step5', 'active');
}

async function analyzeSingle() {
    emptyState.classList.add('hidden');
    resultsArea.classList.remove('hidden');
    analysisProcess.classList.remove('hidden');
    predictionCard.classList.add('hidden');
    document.getElementById('laserScanner').classList.remove('hidden');

    const formData = new FormData();
    formData.append('file', currentFiles[0]);

    try {
        const animPromise = animateSteps();
        const fetchPromise = fetch('/predict', { method:'POST', body:formData }).then(r => r.json());
        
        const [_, data] = await Promise.all([animPromise, fetchPromise]);
        
        setStepState('step5', 'done');
        await new Promise(r => setTimeout(r, 300));
        
        if (data.success) {
            displayResults(data);
        } else {
            alert('Analysis failed: ' + (data.error || 'Unknown error'));
            resetState();
        }
    } catch(err) {
        alert('Could not reach server. Is it running?');
        resetState();
    } finally {
        document.getElementById('laserScanner').classList.add('hidden');
    }
}

// ── Analyze (Bulk) ────────────────────────────────────────────────────────────
async function analyzeBulk() {
    emptyState.classList.add('hidden');
    statsPanel.classList.remove('hidden');
    document.getElementById('generateReportBtn').style.display = 'none';
    document.getElementById('evaluationReport').classList.add('hidden');
    feedbackData = {};
    results = {};
    updateTally();
    bulkResultsArea.classList.remove('hidden');
    bulkProgressContainer.classList.remove('hidden');
    bulkGridContainer.classList.remove('hidden');
    resultsGrid.innerHTML = '';
    
    const total = currentFiles.length;
    
    // 1. Generate Skeleton Loading Squares
    for (let i = 0; i < total; i++) {
        const skeleton = document.createElement('div');
        skeleton.className = 'grid-item skeleton';
        skeleton.id = `item-${i}`;
        resultsGrid.appendChild(skeleton);
    }
    
    // 2. Process Files
    for (let i = 0; i < total; i++) {
        const file = currentFiles[i];
        
        bulkProgressText.textContent = `${i + 1} / ${total}`;
        bulkProgressBar.style.width = `${((i) / total) * 100}%`;
        
        const formData = new FormData();
        formData.append('file', file);
        
        const element = document.getElementById(`item-${i}`);
        
        try {
            const res = await fetch('/predict', { method: 'POST', body: formData });
            const data = await res.json();
            
            element.classList.remove('skeleton');
            
            const isImage = file.type.startsWith('image/');
            const mediaUrl = URL.createObjectURL(file);
            let mediaHtml = isImage 
                ? `<img src="${mediaUrl}" alt="Thumbnail">`
                : `<video src="${mediaUrl}" muted loop onmouseover="this.play()" onmouseout="this.pause()"></video>`;
            mediaHtml += `<div class="feedback-badge" id="badge-${i}"></div>`;
            
            if (data.success) {
                const isFake = data.prediction === 'Fake';
                results[i] = data;
                updateTally();
                element.classList.add(isFake ? 'status-fake' : 'status-real');
                
                element.innerHTML = `
                  ${mediaHtml}
                  <div class="item-details">
                    <div class="detail-name">${file.name}</div>
                    <div class="detail-verdict ${isFake ? 'verdict-fake' : 'verdict-real'}">
                      ${data.prediction.toUpperCase()} (${data.confidence})
                    </div>
                    <div class="metrics-grid">
                      <div>CommFor:</div>
                      <div class="metric-val">${Math.round(data.commfor_score*100)}%</div>
                      <div>B-Free:</div>
                      <div class="metric-val">${Math.round(data.bfree_score*100)}%</div>
                      <div>Prov:</div>
                      <div class="metric-val">${Math.round(data.prov_score*100)}%</div>
                      ${data.gend_score !== null ? `<div>GenD:</div><div class="metric-val">${Math.round(data.gend_score*100)}%</div>` : ''}
                    </div>
                    <div class="feedback-btns">
                      <button class="fb-btn fb-correct" id="fb-correct-${i}" onclick="submitFeedback(${i}, true, event)"><i class="fa-solid fa-check"></i> Correct</button>
                      <button class="fb-btn fb-wrong" id="fb-wrong-${i}" onclick="submitFeedback(${i}, false, event)"><i class="fa-solid fa-xmark"></i> Wrong</button>
                    </div>
                  </div>
                `;
            } else {
                element.classList.add('status-fake');
                element.innerHTML = `
                  ${mediaHtml}
                  <div class="item-details" style="opacity: 1;">
                    <div class="detail-name">${file.name}</div>
                    <div class="detail-verdict verdict-fake">ERROR</div>
                    <div class="metrics-grid">
                      <div style="grid-column: span 2">${data.error || 'Failed'}</div>
                    </div>
                  </div>
                `;
            }
        } catch (err) {
            element.classList.remove('skeleton');
            element.classList.add('status-fake');
            element.innerHTML = `
              <div class="item-details" style="opacity:1;">
                <div class="detail-name">${file.name}</div>
                <div class="detail-verdict verdict-fake">NETWORK ERROR</div>
              </div>
            `;
        }
    }
    
    bulkProgressText.textContent = `${total} / ${total} (Complete)`;
    bulkProgressBar.style.width = `100%`;
    
    analyzeBtn.disabled = false;
    analyzeBtn.classList.remove('disabled');
    
    document.getElementById('generateReportBtn').style.display = 'block';
}

// ── Common Actions ────────────────────────────────────────────────────────────
analyzeBtn.addEventListener('click', async () => {
    if (!currentFiles.length) return;

    analyzeBtn.disabled = true;
    analyzeBtn.classList.add('disabled');

    if (currentFiles.length === 1) {
        await analyzeSingle();
    } else {
        await analyzeBulk();
    }
});

function resetState() {
    analyzeBtn.disabled = false;
    analyzeBtn.classList.remove('disabled');
    analysisProcess.classList.add('hidden');
    resultsArea.classList.add('hidden');
    bulkResultsArea.classList.add('hidden');
}

// ── Display results ────────────────────────────────────────────────────────────
function displayResults(data) {
    analysisProcess.classList.add('hidden');
    predictionCard.classList.remove('hidden');

    const isFake = data.prediction === 'Fake';

    scoreValue.textContent = data.confidence;
    verdict.className = 'verdict ' + (isFake ? 'fake' : 'real');
    verdict.textContent = data.prediction.toUpperCase();

    const col  = isFake ? 'var(--fake-color)' : 'var(--real-color)';
    const glow = isFake ? 'rgba(239,68,68,.4)' : 'rgba(16,185,129,.4)';
    scoreCircle.style.borderColor = col;
    scoreCircle.style.color       = col;
    scoreCircle.style.boxShadow   = `0 0 30px ${glow}`;

    setTimeout(() => { scoreMarker.style.left = `${data.score * 100}%`; }, 100);

    renderDetector('card-commfor', commforScoreEl, data.commfor_score);
    renderDetector('card-bfree',   bfreeScoreEl,   data.bfree_score);
    renderDetector('card-prov',    provScoreEl,    data.prov_score);
    if (data.gend_score !== null) {
        document.getElementById('card-gend').style.display = 'flex';
        renderDetector('card-gend', gendScoreEl, data.gend_score);
    } else {
        document.getElementById('card-gend').style.display = 'none';
    }

    detailsText.textContent =
        `Ensemble score: ${data.score.toFixed(6)}` +
        (data.frames_analyzed > 1 ? ` · ${data.frames_analyzed} frames averaged` : '');

    analyzeBtn.disabled = false;
    analyzeBtn.classList.remove('disabled');
}

function renderDetector(cardId, scoreEl, score) {
    const card     = document.getElementById(cardId);
    const pct      = Math.round(score * 100);
    const isFakeSig = score > 0.30;

    scoreEl.textContent = pct + '%';
    scoreEl.style.color  = isFakeSig ? 'var(--fake-color)' : 'var(--real-color)';
    card.className = 'detector-card ' + (isFakeSig ? 'signal-fake' : 'signal-real');
}

// ── Evaluation & Feedback ──────────────────────────────────────────────────────
function submitFeedback(index, isCorrect, event) {
    event.stopPropagation();
    feedbackData[index] = isCorrect;
    
    const correctBtn = document.getElementById(`fb-correct-${index}`);
    const wrongBtn = document.getElementById(`fb-wrong-${index}`);
    const badge = document.getElementById(`badge-${index}`);
    
    if (isCorrect) {
        correctBtn.classList.add('selected');
        wrongBtn.classList.remove('selected');
        badge.className = 'feedback-badge visible correct';
        badge.innerHTML = '<i class="fa-solid fa-check"></i>';
    } else {
        wrongBtn.classList.add('selected');
        correctBtn.classList.remove('selected');
        badge.className = 'feedback-badge visible wrong';
        badge.innerHTML = '<i class="fa-solid fa-xmark"></i>';
    }
}

function generateEvaluationReport() {
    const totalRated = Object.keys(feedbackData).length;
    if (totalRated === 0) {
        alert("Please rate at least one item using the Correct/Wrong buttons before generating a report.");
        return;
    }
    
    let correctCount = 0;
    for (const key in feedbackData) {
        if (feedbackData[key] === true) correctCount++;
    }
    
    const accuracy = ((correctCount / totalRated) * 100).toFixed(1);
    const stat = (lbl, val, color) => `<div><div class="lbl">${lbl}</div><div class="val" style="color:${color}">${val}</div></div>`;

    // Confusion breakdown: what kind of mistakes does the model make?
    let fakeMissed = 0, realFlagged = 0;
    for (const k in feedbackData) {
        if (feedbackData[k]) continue;
        if (results[k].prediction === 'Real') fakeMissed++; else realFlagged++;
    }

    document.getElementById('evaluationReport').classList.remove('hidden');
    document.getElementById('evalStats').innerHTML =
        stat('Files Rated', totalRated, 'var(--text-primary)') +
        stat('Correct', correctCount, 'var(--real-color)') +
        stat('Fakes missed', fakeMissed, 'var(--fake-color)') +
        stat('Reals flagged', realFlagged, 'var(--fake-color)') +
        `<div class="acc"><div class="lbl">Overall Accuracy</div><div class="val" style="color:${accuracy > 80 ? 'var(--real-color)' : 'var(--fake-color)'}">${accuracy}%</div></div>`;
}

async function calibrateModel() {
    const samples = Object.keys(feedbackData).map(k => {
        const r = results[k];
        const predFake = r.prediction === 'Fake';
        return { commfor: r.commfor_score, bfree: r.bfree_score,
                 is_fake: feedbackData[k] ? predFake : !predFake };
    });
    const msg = document.getElementById('calibrateMsg');
    if (!samples.length) { msg.textContent = 'Rate some items first.'; return; }
    msg.textContent = 'Training...';
    try {
        const res = await (await fetch('/feedback', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(samples) })).json();
        msg.textContent = res.ok
            ? `Saved ${res.stored} labels to the review queue. Live predictions are unchanged; use evaluate.py for real accuracy.`
            : 'Could not save feedback.';
    } catch (e) { msg.textContent = 'Could not reach server.'; }
}

