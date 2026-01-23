// Estado global
let currentSessionId = null;
let websocket = null;
let possessionChart = null;
let passesChart = null;
let timelineChart = null;

// File selection handler
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, initializing file handlers...');
    
    const fileInput = document.getElementById('videoFile');
    const fileInfo = document.getElementById('file-info');
    const fileName = document.getElementById('file-name');
    const fileUploadZone = document.getElementById('fileUploadZone');
    
    console.log('Elements found:', {
        fileInput: !!fileInput,
        fileInfo: !!fileInfo,
        fileName: !!fileName,
        fileUploadZone: !!fileUploadZone
    });
    
    if (!fileInput || !fileUploadZone) {
        console.error('Missing required elements!');
        return;
    }
    
    fileInput.addEventListener('change', function(e) {
        console.log('File selected:', this.files);
        if (this.files && this.files[0]) {
            fileName.textContent = this.files[0].name;
            fileInfo.style.display = 'block';
            console.log('File info displayed');
        }
    });
    
    // Drag and drop
    fileUploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        fileUploadZone.classList.add('dragover');
    });
    
    fileUploadZone.addEventListener('dragleave', () => {
        fileUploadZone.classList.remove('dragover');
    });
    
    fileUploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        fileUploadZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            fileName.textContent = files[0].name;
            fileInfo.style.display = 'block';
            console.log('File dropped:', files[0].name);
        }
    });
    
    console.log('File handlers initialized successfully');
});

// Actualizar placeholder según tipo de URL
function updateUrlPlaceholder() {
    const urlInput = document.getElementById('videoUrl');
    const helpText = document.getElementById('urlHelpText');
    const sourceType = document.getElementById('urlSourceType').value;
    
    const placeholders = {
        'youtube': 'https://www.youtube.com/watch?v=... or https://youtu.be/...',
        'hls': 'https://example.com/stream.m3u8',
        'rtmp': 'rtmp://example.com/live/stream',
        'veo': 'https://veo.co/matches/...'
    };
    
    const helpTexts = {
        'youtube': 'Paste YouTube video URL or live stream link',
        'hls': 'Enter HLS stream URL (.m3u8)',
        'rtmp': 'Enter RTMP stream URL',
        'veo': 'Enter Veo match URL'
    };
    
    urlInput.placeholder = placeholders[sourceType] || placeholders['youtube'];
    helpText.innerHTML = '<i class="fas fa-info-circle"></i> ' + (helpTexts[sourceType] || helpTexts['youtube']);
}

// Analizar desde URL
async function analyzeFromUrl() {
    console.log('analyzeFromUrl called');
    
    const urlInput = document.getElementById('videoUrl');
    const sourceType = document.getElementById('urlSourceType').value;
    const url = urlInput.value.trim();
    
    console.log('URL:', url, 'Type:', sourceType);
    
    if (!url) {
        alert('Please enter a video URL');
        return;
    }
    
    const statusDiv = document.getElementById('upload-status');
    statusDiv.innerHTML = '<div class="alert alert-info"><i class="fas fa-spinner fa-spin me-2"></i>Connecting to stream...</div>';
    
    try {
        console.log('Sending request to /api/analyze/url');
        const response = await fetch('/api/analyze/url', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                url: url,
                source_type: sourceType
            })
        });
        
        const data = await response.json();
        console.log('Response:', data);
        
        if (data.success) {
            currentSessionId = data.session_id;
            statusDiv.innerHTML = '<div class="alert alert-success"><i class="fas fa-check-circle me-2"></i>Stream connected successfully!</div>';
            
            // Conectar WebSocket
            connectWebSocket();
            
            // Mostrar sección de progreso
            document.getElementById('upload-section').style.display = 'none';
            document.getElementById('progress-section').style.display = 'block';
        } else {
            statusDiv.innerHTML = `<div class="alert alert-danger"><i class="fas fa-exclamation-circle me-2"></i>Error: ${data.error}</div>`;
        }
    } catch (error) {
        statusDiv.innerHTML = `<div class="alert alert-danger"><i class="fas fa-exclamation-circle me-2"></i>Error: ${error.message}</div>`;
    }
}

// Subir video
async function uploadVideo() {
    console.log('uploadVideo called');
    
    const fileInput = document.getElementById('videoFile');
    const file = fileInput.files[0];
    
    console.log('File:', file);
    
    if (!file) {
        alert('Por favor selecciona un video');
        return;
    }
    
    const uploadBtn = document.getElementById('uploadBtn');
    const statusDiv = document.getElementById('upload-status');
    
    uploadBtn.disabled = true;
    statusDiv.innerHTML = '<div class="alert alert-info"><i class="fas fa-spinner fa-spin me-2"></i>Subiendo video...</div>';
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        console.log('Uploading file to /api/upload');
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        console.log('Upload response:', data);
        
        if (data.success) {
            currentSessionId = data.session_id;
            statusDiv.innerHTML = '<div class="alert alert-success"><i class="fas fa-check-circle me-2"></i>Video subido correctamente</div>';
            
            // Conectar WebSocket
            connectWebSocket();
            
            // Iniciar análisis
            startAnalysis();
        } else {
            statusDiv.innerHTML = `<div class="alert alert-danger"><i class="fas fa-exclamation-circle me-2"></i>Error: ${data.error}</div>`;
            uploadBtn.disabled = false;
        }
    } catch (error) {
        console.error('Error in uploadVideo:', error);
        statusDiv.innerHTML = `<div class="alert alert-danger"><i class="fas fa-exclamation-circle me-2"></i>Error: ${error.message}</div>`;
        uploadBtn.disabled = false;
    }
}

// Conectar WebSocket
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/${currentSessionId}`;
    
    websocket = new WebSocket(wsUrl);
    
    websocket.onmessage = function(event) {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
    };
    
    websocket.onerror = function(error) {
        console.error('WebSocket error:', error);
    };
    
    websocket.onclose = function() {
        console.log('WebSocket closed');
    };
}

// Manejar mensajes WebSocket
function handleWebSocketMessage(data) {
    console.log('WebSocket message:', data);
    
    if (data.type === 'status') {
        document.getElementById('progress-text').textContent = data.message;
    } else if (data.type === 'progress') {
        updateProgress(data);
    } else if (data.type === 'frame') {
        updateVideoFrame(data);
    } else if (data.type === 'batch_complete') {
        updateBatchComplete(data);
    } else if (data.type === 'completed') {
        showResults(data.stats);
    } else if (data.type === 'error') {
        showError(data.message);
    }
}

// Actualizar frame del video
function updateVideoFrame(data) {
    const canvas = document.getElementById('videoCanvas');
    if (!canvas) {
        console.warn('Canvas element not found');
        return;
    }
    
    const ctx = canvas.getContext('2d');
    
    const img = new Image();
    img.onload = function() {
        // Ajustar tamaño del canvas si es necesario
        if (canvas.width !== img.width || canvas.height !== img.height) {
            canvas.width = img.width;
            canvas.height = img.height;
        }
        
        // Dibujar imagen
        ctx.drawImage(img, 0, 0);
    };
    
    img.onerror = function() {
        console.error('Error loading frame image');
    };
    
    img.src = 'data:image/jpeg;base64,' + data.image;
    
    // Actualizar número de frame si está disponible
    if (data.frame_idx !== undefined) {
        const currentFrameEl = document.getElementById('current-frame');
        if (currentFrameEl) {
            currentFrameEl.textContent = data.frame_idx;
        }
    }
}

// Iniciar análisis
async function startAnalysis() {
    document.getElementById('upload-section').style.display = 'none';
    document.getElementById('progress-section').style.display = 'block';
    
    try {
        const response = await fetch(`/api/analyze/${currentSessionId}`, {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (!data.success) {
            showError(data.error);
        }
    } catch (error) {
        showError(error.message);
    }
}

// Actualizar progreso
function updateProgress(data) {
    const progressBarFill = document.getElementById('progressBarFill');
    const progressPercent = document.getElementById('progressPercent');
    const progressText = document.getElementById('progress-text');
    const currentFrame = document.getElementById('current-frame');
    const totalFrames = document.getElementById('total-frames');
    const currentBatch = document.getElementById('current-batch');
    
    progressBarFill.style.width = data.progress + '%';
    progressPercent.textContent = data.progress + '%';
    
    if (data.frame !== undefined) {
        currentFrame.textContent = data.frame;
    }
    
    if (data.total_frames !== undefined) {
        totalFrames.textContent = data.total_frames;
    }
    
    if (data.batch_idx !== undefined && currentBatch) {
        currentBatch.textContent = data.batch_idx + 1;
    }
    
    progressText.textContent = data.message || `Frame ${data.frame} / ${data.total_frames}`;
}

// Actualizar cuando se completa un batch
function updateBatchComplete(data) {
    const progressText = document.getElementById('progress-text');
    
    // Mostrar mensaje de batch completo
    if (data.message) {
        progressText.textContent = data.message;
    }
    
    // Actualizar estadísticas en tiempo real
    if (data.stats) {
        updateLiveStats(data.stats);
        updateLiveCharts(data.stats);
    }
}

// Actualizar gráficos en tiempo real
function updateLiveCharts(stats) {
    // Inicializar gráficos si no existen
    if (!possessionChart || !passesChart) {
        initializeCharts();
    }
    
    // Actualizar gráfico de posesión
    if (stats.possession_percent && possessionChart) {
        possessionChart.data.datasets[0].data = [
            stats.possession_percent[0] || 0,
            stats.possession_percent[1] || 0
        ];
        possessionChart.update('none');
    }
    
    // Actualizar gráfico de pases
    if (stats.passes && passesChart) {
        passesChart.data.datasets[0].data = [
            stats.passes[0] || 0,
            stats.passes[1] || 0
        ];
        passesChart.update('none');
    }
    
    // Actualizar estadísticas detalladas por equipo (sección LIVE)
    if (stats.possession_percent) {
        const elem0 = document.getElementById('live-possession-percent-0');
        const elem1 = document.getElementById('live-possession-percent-1');
        if (elem0) elem0.textContent = (stats.possession_percent[0] || 0).toFixed(1) + '%';
        if (elem1) elem1.textContent = (stats.possession_percent[1] || 0).toFixed(1) + '%';
        
        const bar0 = document.getElementById('live-possession-bar-0');
        const bar1 = document.getElementById('live-possession-bar-1');
        if (bar0) bar0.style.width = (stats.possession_percent[0] || 0) + '%';
        if (bar1) bar1.style.width = (stats.possession_percent[1] || 0) + '%';
    }
    
    if (stats.possession_seconds) {
        const time0 = document.getElementById('live-possession-time-0');
        const time1 = document.getElementById('live-possession-time-1');
        if (time0) time0.textContent = (stats.possession_seconds[0] || 0).toFixed(1) + 's';
        if (time1) time1.textContent = (stats.possession_seconds[1] || 0).toFixed(1) + 's';
    }
    
    if (stats.passes) {
        const passes0 = document.getElementById('live-passes-0');
        const passes1 = document.getElementById('live-passes-1');
        if (passes0) passes0.textContent = stats.passes[0] || 0;
        if (passes1) passes1.textContent = stats.passes[1] || 0;
    }
    
    // También actualizar las estadísticas finales (para cuando se complete)
    const finalPercent0 = document.getElementById('possession-percent-0');
    const finalPercent1 = document.getElementById('possession-percent-1');
    if (finalPercent0 && stats.possession_percent) finalPercent0.textContent = (stats.possession_percent[0] || 0).toFixed(1) + '%';
    if (finalPercent1 && stats.possession_percent) finalPercent1.textContent = (stats.possession_percent[1] || 0).toFixed(1) + '%';
    
    const finalBar0 = document.getElementById('possession-bar-0');
    const finalBar1 = document.getElementById('possession-bar-1');
    if (finalBar0 && stats.possession_percent) finalBar0.style.width = (stats.possession_percent[0] || 0) + '%';
    if (finalBar1 && stats.possession_percent) finalBar1.style.width = (stats.possession_percent[1] || 0) + '%';
    
    const finalTime0 = document.getElementById('possession-time-0');
    const finalTime1 = document.getElementById('possession-time-1');
    if (finalTime0 && stats.possession_seconds) finalTime0.textContent = (stats.possession_seconds[0] || 0).toFixed(1) + 's';
    if (finalTime1 && stats.possession_seconds) finalTime1.textContent = (stats.possession_seconds[1] || 0).toFixed(1) + 's';
    
    const finalPasses0 = document.getElementById('passes-0');
    const finalPasses1 = document.getElementById('passes-1');
    if (finalPasses0 && stats.passes) finalPasses0.textContent = stats.passes[0] || 0;
    if (finalPasses1 && stats.passes) finalPasses1.textContent = stats.passes[1] || 0;
}

// Actualizar estadísticas en vivo
function updateLiveStats(stats) {
    // Crear o actualizar panel de estadísticas en vivo
    let liveStatsDiv = document.getElementById('live-stats');
    
    if (!liveStatsDiv) {
        // Crear panel si no existe
        const progressSection = document.getElementById('progress-section');
        liveStatsDiv = document.createElement('div');
        liveStatsDiv.id = 'live-stats';
        liveStatsDiv.className = 'mt-4';
        liveStatsDiv.innerHTML = `
            <div class="card">
                <div class="card-header">
                    <h5 class="mb-0"><i class="fas fa-chart-line me-2"></i>Estadísticas en Tiempo Real</h5>
                </div>
                <div class="card-body">
                    <div class="row g-3">
                        <div class="col-md-3">
                            <div class="stat-box">
                                <i class="fas fa-users fa-2x text-primary mb-2"></i>
                                <div class="stat-label">Detecciones</div>
                                <div class="stat-value" id="live-detections">0</div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="stat-box">
                                <i class="fas fa-futbol fa-2x text-success mb-2"></i>
                                <div class="stat-label">Posesión</div>
                                <div class="stat-value" id="live-possession">-</div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="stat-box">
                                <i class="fas fa-bolt fa-2x text-warning mb-2"></i>
                                <div class="stat-label">Eventos</div>
                                <div class="stat-value" id="live-events">0</div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="stat-box">
                                <i class="fas fa-tachometer-alt fa-2x text-info mb-2"></i>
                                <div class="stat-label">FPS Procesado</div>
                                <div class="stat-value" id="live-fps">0</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        progressSection.appendChild(liveStatsDiv);
    }
    
    // Actualizar valores
    if (stats.detections !== undefined) {
        document.getElementById('live-detections').textContent = stats.detections;
    }
    
    if (stats.possession_team !== undefined) {
        const possessionText = stats.possession_team >= 0 ? `Equipo ${stats.possession_team}` : 'Sin posesión';
        document.getElementById('live-possession').textContent = possessionText;
    }
    
    if (stats.events !== undefined) {
        document.getElementById('live-events').textContent = stats.events;
    }
    
    if (stats.fps_processing !== undefined) {
        document.getElementById('live-fps').textContent = stats.fps_processing + ' fps';
    }
}

// Mostrar resultados finales
function showResults(stats) {
    document.getElementById('progress-section').style.display = 'none';
    document.getElementById('results-section').style.display = 'block';
    
    // Stats Overview
    document.getElementById('total-time').textContent = stats.total_seconds.toFixed(1) + 's';
    document.getElementById('total-frames-stat').textContent = stats.total_frames;
    document.getElementById('possession-stat-0').textContent = (stats.possession_percent[0] || 0).toFixed(1) + '%';
    document.getElementById('possession-stat-1').textContent = (stats.possession_percent[1] || 0).toFixed(1) + '%';
    
    // Posesión
    const p0 = stats.possession_percent[0] || 0;
    const p1 = stats.possession_percent[1] || 0;
    
    document.getElementById('possession-percent-0').textContent = p0.toFixed(1) + '%';
    document.getElementById('possession-percent-1').textContent = p1.toFixed(1) + '%';
    
    document.getElementById('possession-bar-0').style.width = p0 + '%';
    document.getElementById('possession-bar-1').style.width = p1 + '%';
    
    // Tiempo
    const t0 = stats.possession_seconds[0] || 0;
    const t1 = stats.possession_seconds[1] || 0;
    
    document.getElementById('possession-time-0').textContent = t0.toFixed(1) + 's';
    document.getElementById('possession-time-1').textContent = t1.toFixed(1) + 's';
    
    // Pases
    const passes0 = stats.passes ? stats.passes[0] || 0 : 0;
    const passes1 = stats.passes ? stats.passes[1] || 0 : 0;
    
    document.getElementById('passes-0').textContent = passes0;
    document.getElementById('passes-1').textContent = passes1;
    
    // Inicializar gráficos
    initializeCharts();
    
    // Actualizar gráficos
    possessionChart.data.datasets[0].data = [p0, p1];
    possessionChart.update();
    
    passesChart.data.datasets[0].data = [
        stats.passes[0] || 0,
        stats.passes[1] || 0
    ];
    passesChart.update();
    
    // Timeline
    if (stats.timeline) {
        updateTimelineChart(stats.timeline, stats.total_frames);
    }
}

// Inicializar gráficos
function initializeCharts() {
    if (possessionChart) return; // Ya inicializados
    
    // Gráfico de posesión (circular)
    const possessionCtx = document.getElementById('possessionChart').getContext('2d');
    possessionChart = new Chart(possessionCtx, {
        type: 'doughnut',
        data: {
            labels: ['Equipo 0', 'Equipo 1'],
            datasets: [{
                data: [0, 0],
                backgroundColor: ['#00c851', '#ff4444'],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'bottom'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.label + ': ' + context.parsed.toFixed(1) + '%';
                        }
                    }
                }
            }
        }
    });
    
    // Gráfico de pases (barras)
    const passesCtx = document.getElementById('passesChart').getContext('2d');
    passesChart = new Chart(passesCtx, {
        type: 'bar',
        data: {
            labels: ['Equipo 0', 'Equipo 1'],
            datasets: [{
                label: 'Pases Completados',
                data: [0, 0],
                backgroundColor: ['#00c851', '#ff4444'],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
    
    // Gráfico de timeline
    const timelineCtx = document.getElementById('timelineChart').getContext('2d');
    timelineChart = new Chart(timelineCtx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Timeline de Posesión',
                data: [],
                backgroundColor: []
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const team = context.dataset.backgroundColor[context.dataIndex] === '#00c851' ? 'Equipo 0' : 'Equipo 1';
                            return team + ': ' + context.parsed.x + ' frames';
                        }
                    }
                }
            },
            scales: {
                x: {
                    stacked: true
                },
                y: {
                    stacked: true
                }
            }
        }
    });
}

// Actualizar timeline
function updateTimelineChart(timeline, totalFrames) {
    const labels = [];
    const data = [];
    const colors = [];
    
    timeline.forEach((segment, i) => {
        const [start, end, team] = segment;
        const duration = end - start;
        
        labels.push(`Segmento ${i + 1}`);
        data.push(duration);
        colors.push(team === 0 ? '#00c851' : '#ff4444');
    });
    
    timelineChart.data.labels = labels;
    timelineChart.data.datasets[0].data = data;
    timelineChart.data.datasets[0].backgroundColor = colors;
    timelineChart.update();
}

// Mostrar error
function showError(message) {
    document.getElementById('progress-section').style.display = 'none';
    document.getElementById('upload-section').style.display = 'block';
    document.getElementById('uploadBtn').disabled = false;
    
    const statusDiv = document.getElementById('upload-status');
    statusDiv.innerHTML = `<div class="alert alert-danger">Error: ${message}</div>`;
}
