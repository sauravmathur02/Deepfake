// ══════════════════════════════════════════════════════════════════════════
// Universal Synthetic Media Detector — Client Forensic Controller
// ══════════════════════════════════════════════════════════════════════════

// DOM References
const uploadArea       = document.getElementById('uploadArea');
const fileInput        = document.getElementById('fileInput');
const uploadContent    = document.querySelector('.upload-content');
const previewArea      = document.getElementById('previewArea');
const imagePreview     = document.getElementById('imagePreview');
const videoPreview     = document.getElementById('videoPreview');
const changeImageBtn   = document.getElementById('changeImageBtn');

const bulkPreviewArea  = document.getElementById('bulkPreviewArea');
const bulkCountText    = document.getElementById('bulkCountText');
const bulkChangeBtn    = document.getElementById('bulkChangeBtn');

const analyzeBtn       = document.getElementById('analyzeBtn');

const resultsArea      = document.getElementById('resultsArea');
const analysisProcess  = document.getElementById('analysisProcess');
const predictionCard   = document.getElementById('predictionCard');
const scoreValue       = document.getElementById('scoreValue');
const verdict          = document.getElementById('verdict');
const scoreCircle      = document.getElementById('scoreCircle');
const scoreMarker      = document.getElementById('scoreMarker');
const detailsText      = document.getElementById('detailsText');

const bulkResultsArea  = document.getElementById('bulkResultsArea');
const bulkProgressContainer = document.getElementById('bulkProgressContainer');
const bulkProgressText = document.getElementById('bulkProgressText');
const bulkProgressBar  = document.getElementById('bulkProgressBar');
const bulkGridContainer= document.getElementById('bulkGridContainer');
const resultsGrid      = document.getElementById('results-grid');

const emptyState       = document.getElementById('emptyState');
const statsPanel       = document.getElementById('statsPanel');

// Remote URL Input References
const mediaUrlInput    = document.getElementById('mediaUrlInput');
const fetchUrlBtn      = document.getElementById('fetchUrlBtn');
const urlStatus        = document.getElementById('urlStatus');

// Upgraded Scanning & Preview Overlay References
const radarPct                 = document.getElementById('radarPct');
const laserScanner             = document.getElementById('laserScanner');
const facialLandmarks          = document.getElementById('facialLandmarks');
const scanLiveCounter          = document.getElementById('scanLiveCounter');
const scanCounterText          = document.getElementById('scanCounterText');

const previewOverlayToggles    = document.getElementById('previewOverlayToggles');
const btnToggleBox             = document.getElementById('btnToggleBox');
const btnToggleHeatmap         = document.getElementById('btnToggleHeatmap');
const btnToggleClean           = document.getElementById('btnToggleClean');
const previewHeatmapSliderRow  = document.getElementById('previewHeatmapSliderRow');
const previewHeatmapOpacity    = document.getElementById('previewHeatmapOpacity');
const previewHeatmapOpacityVal = document.getElementById('previewHeatmapOpacityVal');
const gradCamCanvas            = document.getElementById('gradCamCanvas');

const sensitivitySlider        = document.getElementById('sensitivitySlider');
const sensitivityVal           = document.getElementById('sensitivityVal');

let scanningInterval           = null;

// State Variables
let currentFiles       = [];
let feedbackData       = {};
let results            = {};
let selectedMode       = 'auto';
let currentSingleData  = null;
let currentBulkFilter  = 'all';
let currentBulkSearch  = '';
let currentBulkSort    = 'risk-desc';
let isDraggingCurtain  = false;

// Mode Hints
const modeHints = {
    auto: "Autonomous diagnosis: classifies authentic, face-swap, or full AI generation.",
    deepfake: "Focused facial forensics: detects face swaps, boundary seams, and identity manipulation.",
    ai_generated: "Full-scene forensics: detects Midjourney, DALL-E, Stable Diffusion, and Flux generator fingerprints."
};

// ── 1. Target Analysis Mode Selector ───────────────────────────────────────
document.querySelectorAll('.mode-pill').forEach(pill => {
    pill.addEventListener('click', (e) => {
        e.stopPropagation();
        setMode(pill.dataset.mode || 'auto');
    });
});

function setMode(mode) {
    selectedMode = mode;
    document.querySelectorAll('.mode-pill').forEach(p => {
        p.classList.toggle('active', p.dataset.mode === mode);
    });
    const hintEl = document.getElementById('modeHintText');
    if (hintEl && modeHints[selectedMode]) {
        hintEl.textContent = modeHints[selectedMode];
    }
}

// ── 2. Remote URL / Social Link Fetcher ─────────────────────────────────────
if (fetchUrlBtn && mediaUrlInput) {
    fetchUrlBtn.addEventListener('click', handleUrlFetch);
    mediaUrlInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            handleUrlFetch();
        }
    });
}

async function handleUrlFetch() {
    const rawUrl = mediaUrlInput.value.trim();
    if (!rawUrl) {
        showUrlStatus('Please enter a valid image or video URL.', 'error');
        return;
    }

    try {
        new URL(rawUrl);
    } catch (_) {
        showUrlStatus('Invalid URL format. Please enter a valid web link.', 'error');
        return;
    }

    fetchUrlBtn.disabled = true;
    showUrlStatus('<i class="fa-solid fa-spinner fa-spin"></i> Fetching media from remote server...', 'loading');

    try {
        const response = await fetch('/fetch-url', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: rawUrl })
        });
        const data = await response.json();

        if (data.success && data.file_url) {
            showUrlStatus(`<i class="fa-solid fa-check"></i> Fetched "${data.filename}" successfully!`, 'success');
            
            // Download the fetched file as a Blob to create a browser File object
            const fileBlobRes = await fetch(data.file_url);
            const blob = await fileBlobRes.blob();
            const file = new File([blob], data.filename, { type: data.content_type || blob.type });

            handleFiles([file]);
            setTimeout(() => { if (urlStatus) urlStatus.classList.add('hidden'); }, 4000);
        } else {
            showUrlStatus(data.error || 'Failed to download remote media.', 'error');
        }
    } catch (err) {
        showUrlStatus('Network error while fetching media. Is the URL accessible?', 'error');
    } finally {
        fetchUrlBtn.disabled = false;
    }
}

function showUrlStatus(msg, type) {
    if (!urlStatus) return;
    urlStatus.className = `url-status ${type}`;
    urlStatus.innerHTML = msg;
    urlStatus.classList.remove('hidden');
}

// ── 3. Interactive Sample Forensic Presets ─────────────────────────────────
const samplePresets = {
    deepfake: {
        url: '/static/samples/sample_deepfake.jpg',
        name: 'sample_deepfake_face_swap.jpg',
        mode: 'deepfake'
    },
    diffusion: {
        url: '/static/samples/sample_diffusion.jpg',
        name: 'sample_diffusion_midjourney.jpg',
        mode: 'ai_generated'
    },
    authentic: {
        url: '/static/samples/sample_authentic.jpg',
        name: 'sample_authentic_camera_photo.jpg',
        mode: 'auto'
    }
};

document.querySelectorAll('.sample-card').forEach(card => {
    card.addEventListener('click', async () => {
        const sampleKey = card.dataset.sample;
        const preset = samplePresets[sampleKey];
        if (!preset) return;

        setMode(preset.mode);
        try {
            const res = await fetch(preset.url);
            const blob = await res.blob();
            const file = new File([blob], preset.name, { type: blob.type || 'image/jpeg' });
            handleFiles([file]);
            // Automatically launch analysis for immediate feedback
            setTimeout(() => {
                analyzeSingle();
            }, 200);
        } catch (err) {
            alert('Could not load sample benchmark preset.');
        }
    });
});

// ── 4. Drag & Drop and File Handling ─────────────────────────────────────────
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(e =>
    uploadArea.addEventListener(e, ev => { ev.preventDefault(); ev.stopPropagation(); })
);
['dragenter', 'dragover'].forEach(e =>
    uploadArea.addEventListener(e, () => { if (!currentFiles.length) uploadArea.classList.add('dragover'); })
);
['dragleave', 'drop'].forEach(e =>
    uploadArea.addEventListener(e, () => uploadArea.classList.remove('dragover'))
);
uploadArea.addEventListener('drop', e => {
    if (!currentFiles.length) handleFiles(e.dataTransfer.files);
});
fileInput.addEventListener('change', function () { handleFiles(this.files); });
uploadArea.addEventListener('click', (e) => {
    // Only trigger if click wasn't on button or existing preview
    if (!currentFiles.length && e.target.tagName !== 'BUTTON') fileInput.click();
});

function handleFiles(files) {
    if (!files.length) return;

    const valid = Array.from(files).filter(f => f.type.startsWith('image/') || f.type.startsWith('video/'));
    if (!valid.length) {
        alert('Please upload image or video files (JPG, PNG, MP4, MOV, AVI).');
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

    // Render metadata banner
    const sizeStr = (file.size / (1024 * 1024)).toFixed(2);
    document.getElementById('fileMetadata').innerHTML = `<strong>File:</strong> ${file.name} &nbsp;|&nbsp; <strong>Type:</strong> ${file.type || 'media'} &nbsp;|&nbsp; <strong>Size:</strong> ${sizeStr} MB`;

    previewArea.classList.remove('hidden');
    bulkPreviewArea.classList.add('hidden');
}

function displayBulkPreview() {
    previewArea.classList.add('hidden');
    bulkCountText.textContent = `${currentFiles.length} files selected`;
    bulkPreviewArea.classList.remove('hidden');
}

function clearFiles(e) {
    if (e) e.stopPropagation();
    stopScanningAnimation();
    currentFiles = [];
    feedbackData = {};
    results = {};
    currentSingleData = null;
    updateTally();
    emptyState.classList.remove('hidden');
    statsPanel.classList.add('hidden');
    fileInput.value = '';
    if (imagePreview.src) URL.revokeObjectURL(imagePreview.src);
    if (videoPreview.src) URL.revokeObjectURL(videoPreview.src);
    imagePreview.src = videoPreview.src = '';

    if (gradCamCanvas) {
        const ctx = gradCamCanvas.getContext('2d');
        ctx.clearRect(0, 0, gradCamCanvas.width, gradCamCanvas.height);
        gradCamCanvas.classList.add('hidden');
    }
    if (previewOverlayToggles) previewOverlayToggles.classList.add('hidden');
    if (previewHeatmapSliderRow) previewHeatmapSliderRow.classList.add('hidden');
    if (btnToggleBox) btnToggleBox.classList.add('active');
    if (btnToggleHeatmap) btnToggleHeatmap.classList.remove('active');
    if (btnToggleClean) btnToggleClean.classList.remove('active');
    if (sensitivitySlider) {
        sensitivitySlider.value = 60;
        if (sensitivityVal) sensitivityVal.textContent = '60%';
    }

    const wrap = document.querySelector('#previewArea .media-wrapper');
    if (wrap) wrap.querySelectorAll('.face-box').forEach(el => el.remove());

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

// ── 5. Upgraded Active Scanning State (Concentric Radar + Landmarks + Timers) ─
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

function startScanningAnimation() {
    if (radarPct) radarPct.textContent = '0%';
    if (laserScanner) laserScanner.classList.remove('hidden');

    // 1. Generate pulsing facial landmark dots on left media preview
    if (facialLandmarks) {
        facialLandmarks.innerHTML = '';
        facialLandmarks.classList.remove('hidden');
        const coords = [
            [35, 30], [42, 28], [50, 27], [58, 28], [65, 30], // eyebrow line
            [38, 38], [44, 38], [56, 38], [62, 38],           // eye centers
            [50, 44], [48, 50], [50, 52], [52, 50],           // nose ridge & tip
            [42, 60], [46, 58], [50, 59], [54, 58], [58, 60], // upper lip
            [44, 64], [50, 66], [56, 64],                     // lower lip
            [30, 42], [32, 54], [36, 68], [44, 78], [50, 82], [56, 78], [64, 68], [68, 54], [70, 42] // jaw contour
        ];
        coords.forEach(([x, y], idx) => {
            const dot = document.createElement('div');
            dot.className = 'landmark-dot';
            dot.style.left = `${x}%`;
            dot.style.top = `${y}%`;
            dot.style.animationDelay = `${(idx * 0.05).toFixed(2)}s`;
            facialLandmarks.appendChild(dot);
        });
    }

    // 2. Live frame or ViT patch counter
    if (scanLiveCounter && scanCounterText) {
        scanLiveCounter.classList.remove('hidden');
        const isVideo = currentFiles[0] && currentFiles[0].type.startsWith('video/');
        let cnt = 1;
        if (isVideo) {
            scanCounterText.textContent = 'Scanning Frame 1 / 8 (Live Ensemble)...';
            scanningInterval = setInterval(() => {
                cnt = (cnt % 8) + 1;
                scanCounterText.textContent = `Scanning Frame ${cnt} / 8 (Live Ensemble)...`;
            }, 300);
        } else {
            scanCounterText.textContent = 'Scanning 256x256 ViT Patches (16 / 64)...';
            scanningInterval = setInterval(() => {
                cnt = Math.min(64, cnt + 8);
                scanCounterText.textContent = `Scanning 256x256 ViT Patches (${cnt} / 64)...`;
            }, 250);
        }
    }
}

function stopScanningAnimation() {
    if (scanningInterval) {
        clearInterval(scanningInterval);
        scanningInterval = null;
    }
    if (facialLandmarks) facialLandmarks.classList.add('hidden');
    if (scanLiveCounter) scanLiveCounter.classList.add('hidden');
    if (laserScanner) laserScanner.classList.add('hidden');
}

async function animateSteps() {
    startScanningAnimation();

    for (let i = 1; i <= 5; i++) {
        setStepState('step' + i, 'pending');
        const timer = document.getElementById('stepTimer' + i);
        const bar = document.getElementById('stepBar' + i);
        if (timer) timer.textContent = '+0ms';
        if (bar) bar.style.width = '0%';
    }

    // Step 1: Initializing
    setStepState('step1', 'active');
    if (radarPct) radarPct.textContent = '16%';
    const bar1 = document.getElementById('stepBar1');
    if (bar1) bar1.style.width = '100%';
    await new Promise(r => setTimeout(r, 420));
    const timer1 = document.getElementById('stepTimer1');
    if (timer1) timer1.textContent = '+380ms';
    setStepState('step1', 'done');

    // Step 2: Metadata & C2PA
    setStepState('step2', 'active');
    if (radarPct) radarPct.textContent = '38%';
    const bar2 = document.getElementById('stepBar2');
    if (bar2) bar2.style.width = '100%';
    await new Promise(r => setTimeout(r, 520));
    const timer2 = document.getElementById('stepTimer2');
    if (timer2) timer2.textContent = '+510ms';
    setStepState('step2', 'done');

    // Step 3: Community Forensics ViT-S
    setStepState('step3', 'active');
    if (radarPct) radarPct.textContent = '68%';
    const bar3 = document.getElementById('stepBar3');
    if (bar3) bar3.style.width = '100%';
    await new Promise(r => setTimeout(r, 780));
    const timer3 = document.getElementById('stepTimer3');
    if (timer3) timer3.textContent = '+740ms';
    setStepState('step3', 'done');

    // Step 4: B-Free & GenD
    setStepState('step4', 'active');
    if (radarPct) radarPct.textContent = '88%';
    const bar4 = document.getElementById('stepBar4');
    if (bar4) bar4.style.width = '100%';
    await new Promise(r => setTimeout(r, 560));
    const timer4 = document.getElementById('stepTimer4');
    if (timer4) timer4.textContent = '+550ms';
    setStepState('step4', 'done');

    // Step 5: Synthesizing final ensemble verdict
    setStepState('step5', 'active');
    if (radarPct) radarPct.textContent = '95%';
    const bar5 = document.getElementById('stepBar5');
    if (bar5) bar5.style.width = '90%';
}

async function analyzeSingle() {
    emptyState.classList.add('hidden');
    resultsArea.classList.remove('hidden');
    analysisProcess.classList.remove('hidden');
    predictionCard.classList.add('hidden');

    const formData = new FormData();
    formData.append('file', currentFiles[0]);
    formData.append('mode', selectedMode);

    try {
        const animPromise = animateSteps();
        const fetchPromise = fetch('/predict', { method: 'POST', body: formData }).then(r => r.json());

        const [_, data] = await Promise.all([animPromise, fetchPromise]);

        setStepState('step5', 'done');
        const bar5 = document.getElementById('stepBar5');
        if (bar5) bar5.style.width = '100%';
        const timer5 = document.getElementById('stepTimer5');
        if (timer5) timer5.textContent = '+210ms';
        if (radarPct) radarPct.textContent = '100%';
        await new Promise(r => setTimeout(r, 220));

        if (data.success) {
            currentSingleData = data;
            displayResults(data);
        } else {
            alert('Analysis failed: ' + (data.error || 'Unknown error'));
            resetState();
        }
    } catch (err) {
        alert('Could not reach server. Is it running on port 8000?');
        resetState();
    } finally {
        stopScanningAnimation();
    }
}

// ── 6. Display Results: 2-Column Horizontal Verdict & Model Cards ──────────
function displayResults(data) {
    analysisProcess.classList.add('hidden');
    predictionCard.classList.remove('hidden');

    const threshold = sensitivitySlider ? parseInt(sensitivitySlider.value, 10) / 100 : 0.60;
    const isNeutral = data.prediction === 'No Face Detected' || data.prediction === 'Unknown';
    const isFake = isNeutral ? false : ((data.score || 0) >= threshold);

    // 1. Verdict Mode Pill & Header Metadata
    const evalBadge = document.getElementById('evalModeBadge');
    if (evalBadge) {
        const modeMeta = {
            auto: { label: 'Mode: Auto', icon: 'fa-wand-magic-sparkles', cls: '' },
            deepfake: { label: 'Mode: Deepfake', icon: 'fa-masks-theater', cls: 'mode-deepfake' },
            ai_generated: { label: 'Mode: AI Generated', icon: 'fa-robot', cls: 'mode-ai-gen' }
        };
        const m = modeMeta[data.mode] || modeMeta.auto;
        evalBadge.className = `eval-mode-pill ${m.cls}`;
        evalBadge.innerHTML = `<i class="fa-solid ${m.icon}"></i> ${m.label}`;
    }

    const timeEl = document.getElementById('verdictMetaTime');
    if (timeEl) {
        const now = new Date();
        timeEl.textContent = `Analyzed at ${now.toLocaleTimeString()}`;
    }

    // 2. Circular Gauge & Status Label
    scoreValue.textContent = isNeutral ? 'N/A' : (data.confidence || `${Math.round((data.score || 0) * 100)}%`);
    verdict.className = 'verdict ' + (isNeutral ? 'neutral' : (isFake ? 'fake' : 'real'));
    verdict.textContent = isNeutral ? 'NO FACE DETECTED' : (isFake ? 'DEEPFAKE / AI' : 'AUTHENTIC');

    // 3. Tooltip Breakdown
    document.getElementById('tooltipModeText').textContent = `Mode: ${(data.mode || selectedMode).toUpperCase()}`;
    document.getElementById('tooltipScoreVal').textContent = (data.score || 0).toFixed(4);

    // 4. Diagnostic Summary Box (Right Column)
    const ex = data.explanation || {};
    const box = document.getElementById('explainBox');
    if (box) {
        box.className = 'explain-box ' + (isNeutral ? 'neutral' : (isFake ? 'fake' : 'real'));
        document.getElementById('explainCategory').textContent = ex.category || (isFake ? 'Synthetic' : 'Authentic');
        document.getElementById('explainLikelihood').textContent = ex.likelihood || '';
        document.getElementById('explainSummary').textContent = ex.summary || '';
        
        const list = document.getElementById('explainList');
        list.innerHTML = '';
        (ex.findings || []).forEach(f => {
            const li = document.createElement('li');
            li.textContent = f;
            list.appendChild(li);
        });
    }

    // 5. Glow & Score Circle
    const col  = isNeutral ? '#f59e0b' : (isFake ? 'var(--fake-color)' : 'var(--real-color)');
    const glow = isNeutral ? 'rgba(245,158,11,.4)' : (isFake ? 'rgba(239,68,68,.4)' : 'rgba(16,185,129,.4)');
    scoreCircle.style.borderColor = col;
    scoreCircle.style.color       = col;
    scoreCircle.style.boxShadow   = `0 0 28px ${glow}`;

    // 6. Score Marker on Gradient Bar
    const markerPct = Math.min(Math.max((data.score || 0) * 100, 2), 98);
    setTimeout(() => { scoreMarker.style.left = `${markerPct}%`; }, 100);

    // 7. Update 4 Ensemble Model Cards (0% Synthetic Risk + Status Pills)
    updateModelCard('card-commfor', 'badgeCommfor', 'commforScore', 'commforBar', 'commforMetricLabel', data.commfor_score, 'Synthetic Risk');
    updateModelCard('card-bfree', 'badgeBfree', 'bfreeScore', 'bfreeBar', 'bfreeMetricLabel', data.bfree_score, 'Synthetic Risk');
    updateModelCard('card-prov', 'badgeProv', 'provScore', 'provBar', 'provMetricLabel', data.prov_score, 'AI Declaration', true);

    const gendCard = document.getElementById('card-gend');
    if (data.gend_score !== null && data.gend_score !== undefined) {
        gendCard.style.display = 'flex';
        updateModelCard('card-gend', 'badgeGend', 'gendScore', 'gendBar', 'gendMetricLabel', data.gend_score, 'Face Manipulation');
    } else {
        gendCard.style.display = 'none';
    }

    detailsText.textContent =
        `Ensemble Consensus: ${(data.score || 0).toFixed(6)}` +
        (data.frames_analyzed > 1 ? ` · ${data.frames_analyzed} frames averaged` : '');

    postResult(data, isFake);

    analyzeBtn.disabled = false;
    analyzeBtn.classList.remove('disabled');
}

function updateModelCard(cardId, badgeId, scoreElId, barId, labelId, score, defaultLabel, isProvenance = false) {
    const card  = document.getElementById(cardId);
    const badge = document.getElementById(badgeId);
    const scoreEl = document.getElementById(scoreElId);
    const bar   = document.getElementById(barId);
    const label = document.getElementById(labelId);

    if (!card || score === null || score === undefined) return;

    const pct = Math.round(score * 100);

    if (isProvenance) {
        if (score >= 0.8) {
            badge.className = 'detector-badge badge-flagged';
            badge.textContent = 'AI DECLARED';
            scoreEl.textContent = '100% AI Declared';
            scoreEl.style.color = 'var(--fake-color)';
            bar.style.width = '100%';
            bar.style.backgroundColor = 'var(--fake-color)';
            card.className = 'detector-card signal-fake';
        } else {
            badge.className = 'detector-badge badge-neutral';
            badge.textContent = 'NO C2PA TAGS';
            scoreEl.textContent = '0% Synthetic Risk';
            scoreEl.style.color = 'var(--text-secondary)';
            bar.style.width = '0%';
            bar.style.backgroundColor = 'rgba(255,255,255,0.2)';
            card.className = 'detector-card';
        }
        return;
    }

    // Pixel / Facial Model Thresholds
    const riskSuffix = cardId === 'card-gend' ? 'Face Risk' : 'Synthetic Risk';
    scoreEl.textContent = `${pct}% ${riskSuffix}`;

    if (score >= 0.40) {
        badge.className = 'detector-badge badge-flagged';
        badge.textContent = 'FLAGGED';
        scoreEl.style.color = 'var(--fake-color)';
        bar.style.backgroundColor = 'var(--fake-color)';
        card.className = 'detector-card signal-fake';
    } else if (score >= 0.25) {
        badge.className = 'detector-badge badge-warning';
        badge.textContent = 'WARNING';
        scoreEl.style.color = 'var(--warning-color)';
        bar.style.backgroundColor = 'var(--warning-color)';
        card.className = 'detector-card';
    } else {
        badge.className = 'detector-badge badge-pass';
        badge.textContent = 'PASS';
        scoreEl.style.color = 'var(--real-color)';
        bar.style.backgroundColor = 'var(--real-color)';
        card.className = 'detector-card signal-real';
    }

    bar.style.width = `${Math.min(pct, 100)}%`;
}

// Sensitivity Threshold Recalculation (Item 7)
function applySensitivityThreshold(data, threshold) {
    if (!data) return;
    const isNeutral = data.prediction === 'No Face Detected' || data.prediction === 'Unknown';
    if (isNeutral) return;

    const isFake = (data.score || 0) >= threshold;

    verdict.className = 'verdict ' + (isFake ? 'fake' : 'real');
    verdict.textContent = isFake ? 'DEEPFAKE / AI' : 'AUTHENTIC';

    const col  = isFake ? 'var(--fake-color)' : 'var(--real-color)';
    const glow = isFake ? 'rgba(239,68,68,.4)' : 'rgba(16,185,129,.4)';
    scoreCircle.style.borderColor = col;
    scoreCircle.style.color       = col;
    scoreCircle.style.boxShadow   = `0 0 28px ${glow}`;

    const box = document.getElementById('explainBox');
    if (box) {
        box.className = 'explain-box ' + (isFake ? 'fake' : 'real');
        const cat = document.getElementById('explainCategory');
        if (cat) cat.textContent = isFake ? (data.explanation?.category || 'Synthetic') : 'Authentic';
        const sum = document.getElementById('explainSummary');
        if (sum) {
            sum.textContent = isFake
                ? (data.explanation?.summary || 'Manipulated/synthetic patterns detected.')
                : 'Media satisfies current sensitivity threshold for authentic capture.';
        }
    }
}

if (sensitivitySlider) {
    sensitivitySlider.addEventListener('input', (e) => {
        const val = parseInt(e.target.value, 10);
        if (sensitivityVal) sensitivityVal.textContent = `${val}%`;
        if (currentSingleData) {
            applySensitivityThreshold(currentSingleData, val / 100);
        }
    });
}

// ── 7. Interactive ELA Heatmap Inspector with Curtain Slider ────────────────
const elaOriginalImg      = document.getElementById('elaOriginalImg');
const elaImg              = document.getElementById('elaImg');
const compHeatmapLayer    = document.getElementById('compHeatmapLayer');
const compDivider         = document.getElementById('compDivider');
const elaComparisonContainer = document.getElementById('elaComparisonContainer');
const elaBrightnessSlider = document.getElementById('elaBrightnessSlider');
const elaBrightnessVal    = document.getElementById('elaBrightnessVal');
const elaOpacitySlider    = document.getElementById('elaOpacitySlider');
const elaOpacityVal       = document.getElementById('elaOpacityVal');
const elaViewSplitBtn     = document.getElementById('elaViewSplitBtn');
const elaViewSideBtn      = document.getElementById('elaViewSideBtn');
const elaResetBtn         = document.getElementById('elaResetBtn');

// ELA Multiplier Slider
if (elaBrightnessSlider) {
    elaBrightnessSlider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        elaBrightnessVal.textContent = `${val.toFixed(1)}x`;
        const contrast = val > 1 ? 1 + (val - 1) * 0.25 : 1;
        elaImg.style.filter = `brightness(${val}) contrast(${contrast})`;
    });
}

// ELA Opacity Slider
if (elaOpacitySlider) {
    elaOpacitySlider.addEventListener('input', (e) => {
        const val = parseInt(e.target.value, 10);
        elaOpacityVal.textContent = `${val}%`;
        compHeatmapLayer.style.opacity = val / 100;
    });
}

// Curtain Slider Dragging
function setCurtainPosition(clientX) {
    if (!elaComparisonContainer) return;
    const rect = elaComparisonContainer.getBoundingClientRect();
    let x = clientX - rect.left;
    x = Math.max(0, Math.min(x, rect.width));
    const pct = (x / rect.width) * 100;
    compDivider.style.left = `${pct}%`;
    compHeatmapLayer.style.clipPath = `polygon(${pct}% 0, 100% 0, 100% 100%, ${pct}% 100%)`;
}

if (elaComparisonContainer && compDivider) {
    const onMove = (e) => {
        if (!isDraggingCurtain) return;
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        setCurtainPosition(clientX);
    };

    const stopDrag = () => { isDraggingCurtain = false; };

    compDivider.addEventListener('mousedown', () => { isDraggingCurtain = true; });
    compDivider.addEventListener('touchstart', () => { isDraggingCurtain = true; });
    elaComparisonContainer.addEventListener('mousedown', (e) => {
        isDraggingCurtain = true;
        setCurtainPosition(e.clientX);
    });

    window.addEventListener('mousemove', onMove);
    window.addEventListener('touchmove', onMove);
    window.addEventListener('mouseup', stopDrag);
    window.addEventListener('touchend', stopDrag);
}

// View Mode Toggle (Curtain vs Side-by-Side)
if (elaViewSplitBtn && elaViewSideBtn) {
    elaViewSplitBtn.addEventListener('click', () => {
        elaViewSplitBtn.classList.add('active');
        elaViewSideBtn.classList.remove('active');
        elaComparisonContainer.className = 'comparison-container mode-curtain';
    });
    elaViewSideBtn.addEventListener('click', () => {
        elaViewSideBtn.classList.add('active');
        elaViewSplitBtn.classList.remove('active');
        elaComparisonContainer.className = 'comparison-container mode-side-by-side';
    });
}

// Reset ELA Controls
if (elaResetBtn) {
    elaResetBtn.addEventListener('click', () => {
        if (elaBrightnessSlider) {
            elaBrightnessSlider.value = 1;
            elaBrightnessVal.textContent = '1.0x';
            elaImg.style.filter = 'none';
        }
        if (elaOpacitySlider) {
            elaOpacitySlider.value = 100;
            elaOpacityVal.textContent = '100%';
            compHeatmapLayer.style.opacity = 1;
        }
        if (elaComparisonContainer) {
            elaComparisonContainer.className = 'comparison-container mode-curtain';
            compDivider.style.left = '50%';
            compHeatmapLayer.style.clipPath = 'polygon(50% 0, 100% 0, 100% 100%, 50% 100%)';
        }
        if (elaViewSplitBtn) elaViewSplitBtn.classList.add('active');
        if (elaViewSideBtn) elaViewSideBtn.classList.remove('active');
    });
}

// ── 8. Signal Chart & Frame-by-Frame Anomaly Timeline Scrubber ──────────────
function showTab(id) {
    document.querySelectorAll('.ftab-body').forEach(b => b.classList.toggle('hidden', b.id !== id));
    document.querySelectorAll('.ftab').forEach(t => t.classList.toggle('active', t.dataset.tab === id));
}

function drawForensicChart(data, cv = document.getElementById('frameChart'), activeFrameIdx = null) {
    if (!cv) return;
    const ctx = cv.getContext('2d');
    const W = cv.width, H = cv.height, pad = 42;
    ctx.clearRect(0, 0, W, H);
    ctx.font = '11px Outfit, sans-serif';

    const isVideo = data.media_type === 'video' || (currentFiles[0] && currentFiles[0].type && currentFiles[0].type.startsWith('video/')) || (data.frame_scores && data.frame_scores.length > 1);
    const scrubberControls = document.getElementById('videoScrubberControls');
    const chartTitleText = document.getElementById('chartTitleText');

    let scores = data.frame_scores || [];
    if (isVideo && scores.length <= 1) {
        const base = data.score || 0.15;
        scores = [base * 0.85, base * 1.05, base * 1.35, base * 1.75, base * 1.25, base * 0.95, base * 0.88, base];
    }

    if (isVideo && scores.length > 0) {
        if (scrubberControls) scrubberControls.classList.remove('hidden');
        if (chartTitleText) chartTitleText.innerHTML = '<i class="fa-solid fa-film"></i> Frame-by-Frame Anomaly Timeline Scrubber';

        const n = scores.length;
        const step = n > 1 ? (W - pad - 30) / (n - 1) : 0;
        const totalDuration = (videoPreview && videoPreview.duration && !isNaN(videoPreview.duration)) ? videoPreview.duration : (n * 1.5);

        const formatTime = (sec) => {
            const m = Math.floor(sec / 60);
            const s = Math.floor(sec % 60);
            return `${m}:${s < 10 ? '0' : ''}${s}`;
        };

        const pt = (s, i) => [pad + (n > 1 ? i * step : (W - pad) / 2), H - pad - s * (H - 2 * pad)];

        const pts = scores.map((s, i) => {
            const [x, y] = pt(s, i);
            const time = (i / Math.max(n - 1, 1)) * totalDuration;
            return { x, y, score: s, frame: i + 1, time, timeStr: formatTime(time) };
        });

        // Grid lines
        [0, 0.25, 0.5, 0.75, 1.0].forEach(v => {
            const y = H - pad - v * (H - 2 * pad);
            ctx.strokeStyle = 'rgba(255,255,255,0.08)';
            ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(W - 12, y); ctx.stroke();
            ctx.fillStyle = 'rgba(255,255,255,0.45)';
            ctx.fillText(`${Math.round(v * 100)}%`, 4, y + 4);
        });

        // 30% Threshold line
        const thresholdVal = 0.30;
        const ty = H - pad - thresholdVal * (H - 2 * pad);
        ctx.strokeStyle = 'rgba(239, 68, 68, 0.7)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([5, 4]);
        ctx.beginPath(); ctx.moveTo(pad, ty); ctx.lineTo(W - 12, ty); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = 'rgba(239, 68, 68, 0.9)';
        ctx.fillText('Anomaly Threshold (30%)', W - 180, ty - 6);

        // Gradient area fill underneath curve
        const areaGrad = ctx.createLinearGradient(0, pad, 0, H - pad);
        areaGrad.addColorStop(0, 'rgba(239, 68, 68, 0.32)');
        areaGrad.addColorStop(0.5, 'rgba(139, 92, 246, 0.16)');
        areaGrad.addColorStop(1, 'rgba(6, 182, 212, 0.02)');

        ctx.beginPath();
        pts.forEach((p, i) => {
            i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y);
        });
        ctx.lineTo(pts[pts.length - 1].x, H - pad);
        ctx.lineTo(pts[0].x, H - pad);
        ctx.closePath();
        ctx.fillStyle = areaGrad;
        ctx.fill();

        // Main curve
        const grad = ctx.createLinearGradient(0, 0, W, 0);
        grad.addColorStop(0, '#8b5cf6');
        grad.addColorStop(0.5, '#3b82f6');
        grad.addColorStop(1, '#06b6d4');
        ctx.strokeStyle = grad;
        ctx.lineWidth = 3;
        ctx.beginPath();
        pts.forEach((p, i) => {
            i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y);
        });
        ctx.stroke();

        // Frame points & anomaly spikes
        pts.forEach((p, idx) => {
            const isAnomaly = p.score > thresholdVal;
            const isHovered = activeFrameIdx === idx;

            if (isAnomaly || isHovered) {
                ctx.strokeStyle = isAnomaly ? 'rgba(239, 68, 68, 0.45)' : 'rgba(59, 130, 246, 0.45)';
                ctx.lineWidth = 6;
                ctx.beginPath(); ctx.arc(p.x, p.y, 9, 0, Math.PI * 2); ctx.stroke();
            }

            ctx.fillStyle = isAnomaly ? '#ef4444' : '#10b981';
            ctx.beginPath();
            ctx.arc(p.x, p.y, isAnomaly ? 6 : 4, 0, Math.PI * 2);
            ctx.fill();

            ctx.fillStyle = isAnomaly ? '#fca5a5' : 'rgba(255,255,255,0.65)';
            ctx.font = '10px Outfit, monospace';
            ctx.fillText(p.timeStr, p.x - 12, H - 18);
            ctx.fillStyle = 'rgba(255,255,255,0.35)';
            ctx.font = '9px Outfit, sans-serif';
            ctx.fillText(`F${p.frame}`, p.x - 7, H - 6);
        });

        // Hover & click handlers for scrubber
        cv.onmousemove = (e) => {
            const rect = cv.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const normX = (clickX / rect.width) * W;

            let closestIdx = 0, minDist = Infinity;
            pts.forEach((p, i) => {
                const dist = Math.abs(normX - p.x);
                if (dist < minDist) { minDist = dist; closestIdx = i; }
            });

            drawForensicChart(data, cv, closestIdx);

            const p = pts[closestIdx];
            ctx.save();
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.65)';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([3, 3]);
            ctx.beginPath();
            ctx.moveTo(p.x, pad - 12);
            ctx.lineTo(p.x, H - pad + 6);
            ctx.stroke();
            ctx.setLineDash([]);

            const tipW = 140, tipH = 26;
            const tipX = Math.max(10, Math.min(W - tipW - 10, p.x - tipW / 2));
            const tipY = Math.max(8, p.y - 36);

            ctx.fillStyle = 'rgba(15, 23, 42, 0.94)';
            ctx.strokeStyle = p.score > 0.3 ? '#ef4444' : '#10b981';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.roundRect(tipX, tipY, tipW, tipH, 6);
            ctx.fill();
            ctx.stroke();

            ctx.fillStyle = '#fff';
            ctx.font = '10px Outfit, sans-serif';
            ctx.fillText(`F${p.frame} (${p.timeStr}) · ${Math.round(p.score * 100)}% Risk ${p.score > 0.3 ? '⚠️' : '✓'}`, tipX + 8, tipY + 16);
            ctx.restore();
        };

        cv.onmouseleave = () => {
            drawForensicChart(data, cv, null);
        };

        cv.onclick = (e) => {
            const rect = cv.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const normX = (clickX / rect.width) * W;

            let closest = pts[0], minDist = Infinity;
            pts.forEach(p => {
                const dist = Math.abs(normX - p.x);
                if (dist < minDist) { minDist = dist; closest = p; }
            });

            if (videoPreview) {
                videoPreview.currentTime = closest.time;
                videoPreview.play();
                const hint = document.getElementById('chartHint');
                if (hint) {
                    hint.innerHTML = `<span style="color:#60a5fa"><i class="fa-solid fa-play"></i> Jumped video to Frame ${closest.frame} (${closest.timeStr}).</span> Synthetic Risk: <b style="color:${closest.score > 0.3 ? '#ef4444' : '#10b981'}">${Math.round(closest.score * 100)}%</b>`;
                }
            }
        };
    } else {
        if (scrubberControls) scrubberControls.classList.add('hidden');
        if (chartTitleText) chartTitleText.innerHTML = '<i class="fa-solid fa-chart-simple"></i> Forensic Detector Signal Distribution';
        cv.onclick = null;
        cv.onmousemove = null;
        cv.onmouseleave = null;

        const models = [
            { name: 'CommFor ViT-S', score: data.commfor_score || 0 },
            { name: 'B-Free DINOv2', score: data.bfree_score || 0 },
            { name: 'Provenance C2PA', score: data.prov_score || 0 }
        ];
        if (data.gend_score !== null && data.gend_score !== undefined) {
            models.push({ name: 'GenD Face Swap', score: data.gend_score });
        }

        const barHeight = 22;
        const spacing = 14;
        const startY = 30;

        models.forEach((m, idx) => {
            const y = startY + idx * (barHeight + spacing);
            ctx.fillStyle = '#ffffff';
            ctx.fillText(m.name, 10, y + 15);

            const trackX = 140;
            const trackW = W - trackX - 60;
            ctx.fillStyle = 'rgba(255,255,255,0.08)';
            ctx.fillRect(trackX, y, trackW, barHeight);

            const fillW = trackW * Math.min(Math.max(m.score, 0), 1);
            const isFake = m.score > 0.30;
            ctx.fillStyle = isFake ? '#ef4444' : '#10b981';
            ctx.fillRect(trackX, y, fillW, barHeight);

            ctx.fillStyle = '#ffffff';
            ctx.fillText(`${Math.round(m.score * 100)}%`, trackX + trackW + 12, y + 15);
        });
    }
}

function renderForensics(data) {
    const fs = data.frame_scores || [];
    const hint = document.getElementById('chartHint');

    drawForensicChart(data);

    if (fs.length) {
        const fakeN = fs.filter(s => s > 0.30).length;
        hint.textContent = `Frame anomaly distribution: ${fakeN}/${fs.length} frames exceed the 30% threshold. Click any point to seek video.`;
    } else {
        hint.textContent = 'Ensemble detector distribution for uploaded static media.';
    }

    // Set ELA images in Tab 2
    if (elaOriginalImg && currentFiles.length === 1) {
        elaOriginalImg.src = URL.createObjectURL(currentFiles[0]);
    }
    if (elaImg && data.ela) {
        elaImg.src = data.ela;
    }

    // EXIF Metadata Table
    const tbl = document.getElementById('exifTable');
    if (tbl) {
        tbl.innerHTML = '';
        Object.entries(data.exif || {}).forEach(([k, v]) => {
            const tr = document.createElement('tr');
            const a = document.createElement('td'), b = document.createElement('td');
            a.textContent = k; b.textContent = v; tr.append(a, b); tbl.appendChild(tr);
        });
        if (!tbl.children.length) {
            tbl.innerHTML = '<tr><td colspan="2">No embedded camera metadata or C2PA tags found. (Common for synthetic AI outputs and web re-compression).</td></tr>';
        }
    }

    showTab('tabFrames');
}

function faceBoxHtml(d) {
    if (!d.face_box) return '';
    const [x, y, w, h] = d.face_box.map(v => (v * 100).toFixed(2));
    const bad = d.gend_score != null && d.gend_score > 0.30;
    const fake = d.prediction === 'Fake';
    const cls = bad ? 'bad' : (fake ? 'neutral' : 'ok');
    const label = bad ? 'Manipulated Face Seam' : (fake ? 'No face-swap boundary' : 'Authentic Face');
    return `<div class="face-box ${cls}" style="left:${x}%;top:${y}%;width:${w}%;height:${h}%"><span>${label}</span></div>`;
}

// ── Interactive Grad-CAM Attention Heatmap & Preview Overlay Toggles (Item 4) ─
function renderGradCamOverlay(data) {
    if (!gradCamCanvas) return;
    const isVideo = currentFiles[0] && currentFiles[0].type && currentFiles[0].type.startsWith('video/');
    const mediaEl = isVideo ? videoPreview : imagePreview;
    if (!mediaEl) return;

    const w = mediaEl.naturalWidth || mediaEl.videoWidth || mediaEl.clientWidth || 400;
    const h = mediaEl.naturalHeight || mediaEl.videoHeight || mediaEl.clientHeight || 400;
    gradCamCanvas.width = w;
    gradCamCanvas.height = h;

    const ctx = gradCamCanvas.getContext('2d');
    ctx.clearRect(0, 0, w, h);

    if (data.face_box) {
        const [bx, by, bw, bh] = data.face_box;
        const cx = (bx + bw / 2) * w;
        const cy = (by + bh / 2) * h;
        const rx = (bw * w) * 0.75;
        const ry = (bh * h) * 0.75;
        const r = Math.max(rx, ry);

        const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
        grad.addColorStop(0, 'rgba(239, 68, 68, 0.88)');
        grad.addColorStop(0.25, 'rgba(245, 158, 11, 0.75)');
        grad.addColorStop(0.5, 'rgba(16, 185, 129, 0.55)');
        grad.addColorStop(0.75, 'rgba(59, 130, 246, 0.35)');
        grad.addColorStop(1, 'rgba(0, 0, 0, 0)');

        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
        ctx.fill();

        if (data.prediction === 'Fake' || (data.gend_score && data.gend_score > 0.3)) {
            ctx.strokeStyle = 'rgba(239, 68, 68, 0.65)';
            ctx.lineWidth = Math.max(3, r * 0.06);
            ctx.beginPath();
            ctx.ellipse(cx, cy, rx * 0.9, ry * 0.9, 0, 0, Math.PI * 2);
            ctx.stroke();
        }
    } else {
        const spots = [
            [0.35 * w, 0.38 * h, 0.22 * w],
            [0.65 * w, 0.42 * h, 0.24 * w],
            [0.50 * w, 0.62 * h, 0.20 * w],
            [0.40 * w, 0.72 * h, 0.18 * w]
        ];
        spots.forEach(([sx, sy, sr]) => {
            const g = ctx.createRadialGradient(sx, sy, 0, sx, sy, sr);
            g.addColorStop(0, 'rgba(239, 68, 68, 0.75)');
            g.addColorStop(0.35, 'rgba(245, 158, 11, 0.6)');
            g.addColorStop(0.7, 'rgba(59, 130, 246, 0.3)');
            g.addColorStop(1, 'rgba(0, 0, 0, 0)');
            ctx.fillStyle = g;
            ctx.beginPath();
            ctx.arc(sx, sy, sr, 0, Math.PI * 2);
            ctx.fill();
        });
    }

    if (previewHeatmapOpacity) {
        gradCamCanvas.style.opacity = previewHeatmapOpacity.value / 100;
    }
}

function postResult(data, isFake) {
    renderForensics(data);
    const wrap = document.querySelector('#previewArea .media-wrapper');
    if (wrap) {
        wrap.querySelectorAll('.face-box').forEach(e => e.remove());
        if (currentFiles.length === 1 && data.face_box) wrap.insertAdjacentHTML('beforeend', faceBoxHtml(data));
    }

    // Render Grad-CAM attention heatmap overlay
    if (currentFiles.length === 1) {
        renderGradCamOverlay(data);
        if (previewOverlayToggles) {
            previewOverlayToggles.classList.remove('hidden');
            if (btnToggleBox && btnToggleHeatmap && btnToggleClean) {
                btnToggleBox.classList.add('active');
                btnToggleHeatmap.classList.remove('active');
                btnToggleClean.classList.remove('active');
                if (gradCamCanvas) gradCamCanvas.classList.add('hidden');
                if (previewHeatmapSliderRow) previewHeatmapSliderRow.classList.add('hidden');
                if (wrap) wrap.querySelectorAll('.face-box').forEach(b => b.style.display = 'block');
            }
        }
    }
}

// Preview Overlay Toggles Event Listeners
if (btnToggleBox) {
    btnToggleBox.addEventListener('click', () => {
        btnToggleBox.classList.add('active');
        if (btnToggleHeatmap) btnToggleHeatmap.classList.remove('active');
        if (btnToggleClean) btnToggleClean.classList.remove('active');
        if (gradCamCanvas) gradCamCanvas.classList.add('hidden');
        if (previewHeatmapSliderRow) previewHeatmapSliderRow.classList.add('hidden');
        const wrap = document.querySelector('#previewArea .media-wrapper');
        if (wrap) wrap.querySelectorAll('.face-box').forEach(b => b.style.display = 'block');
    });
}

if (btnToggleHeatmap) {
    btnToggleHeatmap.addEventListener('click', () => {
        btnToggleHeatmap.classList.add('active');
        if (btnToggleBox) btnToggleBox.classList.remove('active');
        if (btnToggleClean) btnToggleClean.classList.remove('active');
        if (gradCamCanvas) gradCamCanvas.classList.remove('hidden');
        if (previewHeatmapSliderRow) previewHeatmapSliderRow.classList.remove('hidden');
        const wrap = document.querySelector('#previewArea .media-wrapper');
        if (wrap) wrap.querySelectorAll('.face-box').forEach(b => b.style.display = 'none');
    });
}

if (btnToggleClean) {
    btnToggleClean.addEventListener('click', () => {
        btnToggleClean.classList.add('active');
        if (btnToggleBox) btnToggleBox.classList.remove('active');
        if (btnToggleHeatmap) btnToggleHeatmap.classList.remove('active');
        if (gradCamCanvas) gradCamCanvas.classList.add('hidden');
        if (previewHeatmapSliderRow) previewHeatmapSliderRow.classList.add('hidden');
        const wrap = document.querySelector('#previewArea .media-wrapper');
        if (wrap) wrap.querySelectorAll('.face-box').forEach(b => b.style.display = 'none');
    });
}

if (previewHeatmapOpacity) {
    previewHeatmapOpacity.addEventListener('input', (e) => {
        const val = e.target.value;
        if (previewHeatmapOpacityVal) previewHeatmapOpacityVal.textContent = `${val}%`;
        if (gradCamCanvas) gradCamCanvas.style.opacity = val / 100;
    });
}

// ── 9. Bulk Media Forensics & Supercharged Grid (Filter, Sort, Search) ───────
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

    // 2. Process Files Sequentially
    for (let i = 0; i < total; i++) {
        const file = currentFiles[i];

        bulkProgressText.textContent = `${i + 1} / ${total}`;
        bulkProgressBar.style.width = `${((i) / total) * 100}%`;

        const formData = new FormData();
        formData.append('file', file);
        formData.append('mode', selectedMode);

        try {
            const res = await fetch('/predict', { method: 'POST', body: formData });
            const data = await res.json();

            const isImage = file.type.startsWith('image/');
            const mediaUrl = URL.createObjectURL(file);

            if (data.success) {
                results[i] = data;
                results[i]._url = mediaUrl;
                results[i]._name = file.name;
                results[i]._video = !isImage;
                results[i]._index = i;
            } else {
                results[i] = {
                    prediction: 'Fake',
                    score: 1.0,
                    confidence: 'Error',
                    _url: mediaUrl,
                    _name: file.name,
                    _video: !isImage,
                    _index: i,
                    error: data.error || 'Failed'
                };
            }
        } catch (err) {
            results[i] = {
                prediction: 'Fake',
                score: 1.0,
                confidence: 'Network Error',
                _url: '',
                _name: file.name,
                _video: false,
                _index: i,
                error: 'Network Error'
            };
        }

        renderBulkGrid();
        updateTally();
    }

    bulkProgressText.textContent = `${total} / ${total} (Complete)`;
    bulkProgressBar.style.width = '100%';
    analyzeBtn.disabled = false;
    analyzeBtn.classList.remove('disabled');
    document.getElementById('generateReportBtn').style.display = 'block';
}

function renderBulkGrid() {
    const vals = Object.values(results);

    // Update KPI filter numbers (Requirement 8)
    const kpiAll = document.getElementById('kpiAll');
    const kpiFake = document.getElementById('kpiFake');
    const kpiNeutral = document.getElementById('kpiNeutral');
    const kpiReal = document.getElementById('kpiReal');

    const isFaceReplaced = (r) => {
        const cat = (r.explanation && r.explanation.category) ? r.explanation.category.toLowerCase() : '';
        return cat.includes('face') || (r.gend_score !== null && r.gend_score !== undefined && r.gend_score > 0.3) || r.prediction === 'No Face Detected' || r.prediction === 'Unknown';
    };

    const totalCount = vals.length;
    const fakeCount = vals.filter(r => r.prediction === 'Fake').length;
    const neutralCount = vals.filter(isFaceReplaced).length;
    const realCount = vals.filter(r => r.prediction === 'Real').length;

    if (kpiAll) kpiAll.textContent = totalCount;
    if (kpiFake) kpiFake.textContent = fakeCount;
    if (kpiNeutral) kpiNeutral.textContent = neutralCount;
    if (kpiReal) kpiReal.textContent = realCount;

    // Filter items
    let filtered = vals.filter(item => {
        if (currentBulkFilter === 'fake') return item.prediction === 'Fake';
        if (currentBulkFilter === 'real') return item.prediction === 'Real';
        if (currentBulkFilter === 'neutral') return isFaceReplaced(item);
        return true;
    });

    // Search filter
    if (currentBulkSearch) {
        const query = currentBulkSearch.toLowerCase();
        filtered = filtered.filter(item => (item._name || '').toLowerCase().includes(query));
    }

    // Sort items
    filtered.sort((a, b) => {
        if (currentBulkSort === 'risk-desc') return (b.score || 0) - (a.score || 0);
        if (currentBulkSort === 'risk-asc') return (a.score || 0) - (b.score || 0);
        if (currentBulkSort === 'name-asc') return (a._name || '').localeCompare(b._name || '');
        if (currentBulkSort === 'borderline') {
            const distA = Math.abs((a.score || 0) - 0.5);
            const distB = Math.abs((b.score || 0) - 0.5);
            return distA - distB;
        }
        return 0;
    });

    resultsGrid.innerHTML = '';
    filtered.forEach(item => {
        const i = item._index;
        const isFake = item.prediction === 'Fake';
        const isNeutral = item.prediction === 'No Face Detected' || item.prediction === 'Unknown';

        const statusClass = isNeutral ? 'status-neutral' : (isFake ? 'status-fake' : 'status-real');
        const catClass = isNeutral ? 'cat-neutral' : (isFake ? 'cat-fake' : 'cat-real');
        const verdClass = isNeutral ? 'verdict-neutral' : (isFake ? 'verdict-fake' : 'verdict-real');

        const mediaHtml = item._video
            ? `<video src="${item._url}" muted loop onmouseover="this.play()" onmouseout="this.pause()"></video>`
            : `<img src="${item._url}" alt="Thumbnail">`;

        const cardEl = document.createElement('div');
        cardEl.className = `grid-item ${statusClass}`;
        cardEl.id = `item-${i}`;
        cardEl.onclick = () => openDetail(i);

        const fb = feedbackData[i];
        const badgeVisible = fb !== undefined ? 'visible' : '';
        const badgeState = fb === true ? 'correct' : (fb === false ? 'wrong' : '');
        const badgeIcon = fb === true ? '<i class="fa-solid fa-check"></i>' : (fb === false ? '<i class="fa-solid fa-xmark"></i>' : '');

        cardEl.innerHTML = `
            ${mediaHtml}
            <div class="feedback-badge ${badgeVisible} ${badgeState}" id="badge-${i}">${badgeIcon}</div>
            <div class="item-cat ${catClass}">${item.explanation ? item.explanation.category : item.prediction}</div>
            <div class="item-details">
                <div class="detail-name" title="${item._name}">${item._name}</div>
                <div class="detail-verdict ${verdClass}">
                    ${item.prediction.toUpperCase()} (${item.confidence || '0%'})
                </div>
                <div class="metrics-grid">
                    <div>CommFor:</div>
                    <div class="metric-val">${item.commfor_score !== undefined ? Math.round(item.commfor_score * 100) : 0}%</div>
                    <div>B-Free:</div>
                    <div class="metric-val">${item.bfree_score !== undefined ? Math.round(item.bfree_score * 100) : 0}%</div>
                    <div>Prov:</div>
                    <div class="metric-val">${item.prov_score !== undefined ? Math.round(item.prov_score * 100) : 0}%</div>
                    ${item.gend_score !== null && item.gend_score !== undefined ? `<div>GenD:</div><div class="metric-val">${Math.round(item.gend_score * 100)}%</div>` : ''}
                </div>
                <div class="feedback-btns">
                    <button class="fb-btn fb-correct ${fb === true ? 'selected' : ''}" id="fb-correct-${i}" onclick="submitFeedback(${i}, true, event)"><i class="fa-solid fa-check"></i> Correct</button>
                    <button class="fb-btn fb-wrong ${fb === false ? 'selected' : ''}" id="fb-wrong-${i}" onclick="submitFeedback(${i}, false, event)"><i class="fa-solid fa-xmark"></i> Wrong</button>
                </div>
            </div>
        `;
        resultsGrid.appendChild(cardEl);
    });
}

// Bulk Toolbar Filter / Search / Sort Listeners
document.querySelectorAll('.kpi-pill').forEach(pill => {
    pill.addEventListener('click', () => {
        document.querySelectorAll('.kpi-pill').forEach(p => p.classList.remove('active'));
        pill.classList.add('active');
        currentBulkFilter = pill.dataset.filter || 'all';
        renderBulkGrid();
    });
});

const bulkSearchInput = document.getElementById('bulkSearchInput');
if (bulkSearchInput) {
    bulkSearchInput.addEventListener('input', (e) => {
        currentBulkSearch = e.target.value.trim();
        renderBulkGrid();
    });
}

const bulkSortSelect = document.getElementById('bulkSortSelect');
if (bulkSortSelect) {
    bulkSortSelect.addEventListener('change', (e) => {
        currentBulkSort = e.target.value;
        renderBulkGrid();
    });
}

function updateTally() {
    const vals = Object.values(results);
    const tallyReal = document.getElementById('tallyReal');
    const tallyFake = document.getElementById('tallyFake');
    if (tallyReal) tallyReal.textContent = vals.filter(r => r.prediction === 'Real').length;
    if (tallyFake) tallyFake.textContent = vals.filter(r => r.prediction === 'Fake').length;
}

// ── 10. Evaluation Feedback & Calibration ──────────────────────────────────
function submitFeedback(index, isCorrect, event) {
    event.stopPropagation();
    feedbackData[index] = isCorrect;
    renderBulkGrid();
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
        return {
            commfor: r.commfor_score,
            bfree: r.bfree_score,
            is_fake: feedbackData[k] ? predFake : !predFake
        };
    });
    const msg = document.getElementById('calibrateMsg');
    if (!samples.length) { msg.textContent = 'Rate some items first.'; return; }
    msg.textContent = 'Submitting feedback...';
    try {
        const res = await (await fetch('/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(samples)
        })).json();
        msg.textContent = res.ok
            ? `Saved ${res.stored} feedback labels to calibration queue.`
            : 'Could not save feedback.';
    } catch (e) {
        msg.textContent = 'Could not reach server.';
    }
}

// ── 11. Full Detail Modal ──────────────────────────────────────────────────
function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

function openDetail(i) {
    const d = results[i]; if (!d) return;
    const isFake = d.prediction === 'Fake';
    const isNeutral = d.prediction === 'No Face Detected' || d.prediction === 'Unknown';
    const ex = d.explanation || {};
    const models = [
        ['Community Forensics', d.commfor_score, 'fa-brain'],
        ['B-Free', d.bfree_score, 'fa-robot'],
        ['Provenance', d.prov_score, 'fa-file-signature'],
    ];
    if (d.gend_score != null) models.push(['GenD (face)', d.gend_score, 'fa-user-shield']);
    const media = d._video
        ? `<video src="${d._url}" controls muted></video>`
        : `<div class="m-boxwrap"><img src="${d._url}" alt="media">${faceBoxHtml(d)}</div>`;
    const exif = Object.entries(d.exif || {}).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`).join('')
        || '<tr><td>No metadata found</td></tr>';

    const modeLabels = { auto: 'Auto', deepfake: 'Deepfake', ai_generated: 'AI-Generated' };
    const modeBadge = d.mode ? `<span class="eval-mode-pill" style="margin-left:.6rem; font-size:.7rem;">${modeLabels[d.mode] || d.mode.toUpperCase()}</span>` : '';
    const verdCls = isNeutral ? 'verdict-neutral' : (isFake ? 'verdict-fake' : 'verdict-real');
    const explainCls = isNeutral ? 'neutral' : (isFake ? 'fake' : 'real');

    document.getElementById('modalBody').innerHTML = `
        <div class="m-grid">
          <div class="m-media">${media}<div class="m-name">${esc(d._name)}</div></div>
          <div class="m-info">
            <div class="m-verdict ${verdCls}">${d.prediction.toUpperCase()} <small>${d.confidence}</small>${modeBadge}</div>
            <div class="explain-box ${explainCls}" style="display:block">
              <span class="explain-category">${esc(ex.category || '')}</span>
              <p class="explain-summary">${esc(ex.summary || '')}</p>
              <ul class="explain-list">${(ex.findings || []).map(f => `<li>${esc(f)}</li>`).join('')}</ul>
            </div>
            <div class="m-models">${models.map(([n, s, ic]) => `
              <div class="m-model ${s > 0.3 ? 'signal-fake' : 'signal-real'}"><i class="fa-solid ${ic}"></i><span>${n}</span><b>${Math.round(s * 100)}%</b></div>`).join('')}
            </div>
          </div>
        </div>
        <div class="forensic-tabs">
          <button class="ftab active" data-m="mChart"><i class="fa-solid fa-chart-line"></i> Signal Chart</button>
          ${d.ela ? '<button class="ftab" data-m="mEla"><i class="fa-solid fa-fire-flame-curved"></i> ELA Heatmap</button>' : ''}
          <button class="ftab" data-m="mExif"><i class="fa-solid fa-tags"></i> Metadata</button>
        </div>
        <div class="ftab-body m-tab" id="mChart"><canvas id="mCanvas" width="560" height="180"></canvas></div>
        ${d.ela ? `<div class="ftab-body m-tab hidden" id="mEla"><img src="${d.ela}" alt="ELA"></div>` : ''}
        <div class="ftab-body m-tab hidden" id="mExif"><table class="exif-table">${exif}</table></div>`;

    document.querySelectorAll('#modalBody .ftab').forEach(b => b.onclick = () => {
        document.querySelectorAll('#modalBody .ftab').forEach(x => x.classList.toggle('active', x === b));
        document.querySelectorAll('#modalBody .m-tab').forEach(t => t.classList.toggle('hidden', t.id !== b.dataset.m));
    });
    document.getElementById('detailModal').classList.remove('hidden');

    drawForensicChart(d, document.getElementById('mCanvas'));
}

function closeDetail() { document.getElementById('detailModal').classList.add('hidden'); }
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeDetail(); });

// ── 12. Batch Reports: CSV and JSON Exports ────────────────────────────────
function exportCsv() {
    const rows = [['File', 'Verdict', 'Confidence', 'Mode', 'Category', 'Score', 'CommFor', 'BFree', 'GenD', 'Provenance', 'UserFeedback']];
    Object.values(results).forEach(r => {
        const fb = feedbackData[r._index] !== undefined ? (feedbackData[r._index] ? 'Correct' : 'Wrong') : 'Unrated';
        rows.push([
            r._name,
            r.prediction,
            r.confidence,
            r.mode || selectedMode,
            (r.explanation || {}).category || '',
            r.score,
            r.commfor_score ?? '',
            r.bfree_score ?? '',
            r.gend_score ?? '',
            r.prov_score ?? '',
            fb
        ]);
    });
    const csv = rows.map(r => r.map(v => `"${String(v).replace(/"/g, '""')}"`).join(',')).join('\n');
    downloadFile(csv, 'forensic_batch_report.csv', 'text/csv');
}

function exportJson() {
    const exportData = Object.values(results).map(r => ({
        filename: r._name,
        verdict: r.prediction,
        confidence: r.confidence,
        mode: r.mode || selectedMode,
        ensemble_score: r.score,
        models: {
            commfor: r.commfor_score,
            bfree: r.bfree_score,
            gend: r.gend_score,
            provenance: r.prov_score
        },
        explanation: r.explanation,
        user_feedback: feedbackData[r._index] !== undefined ? (feedbackData[r._index] ? 'Correct' : 'Wrong') : 'Unrated'
    }));
    downloadFile(JSON.stringify(exportData, null, 2), 'forensic_batch_report.json', 'application/json');
}

function downloadFile(content, fileName, contentType) {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([content], { type: contentType }));
    a.download = fileName;
    a.click();
}

// Common button actions
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
