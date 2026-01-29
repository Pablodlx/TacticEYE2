// Estado global
let currentSessionId = null;
let websocket = null;
let possessionChart = null;
let passesChart = null;
let timelineChart = null;

// Función para reiniciar la interfaz
function resetInterface() {
    // Cerrar WebSocket si existe
    if (websocket) {
        websocket.close();
        websocket = null;
    }
    
    // Resetear session ID
    currentSessionId = null;
    
    // Limpiar gráficos
    if (possessionChart) {
        possessionChart.destroy();
        possessionChart = null;
    }
    if (passesChart) {
        passesChart.destroy();
        passesChart = null;
    }
    if (timelineChart) {
        timelineChart.destroy();
        timelineChart = null;
    }
    
    // Mostrar sección de upload, ocultar progreso y resultados
    document.getElementById('upload-section').style.display = 'block';
    document.getElementById('progress-section').style.display = 'none';
    document.getElementById('results-section').style.display = 'none';
    
    // Limpiar formularios
    const fileInput = document.getElementById('videoFile');
    if (fileInput) fileInput.value = '';
    
    const urlInput = document.getElementById('videoUrl');
    if (urlInput) urlInput.value = '';
    
    const fileInfo = document.getElementById('file-info');
    if (fileInfo) fileInfo.style.display = 'none';
    
    const statusDiv = document.getElementById('upload-status');
    if (statusDiv) statusDiv.innerHTML = '';
    
    // Limpiar canvas de video
    const videoCanvas = document.getElementById('video-canvas');
    if (videoCanvas) {
        const ctx = videoCanvas.getContext('2d');
        ctx.clearRect(0, 0, videoCanvas.width, videoCanvas.height);
    }
    
    // Limpiar progress bar
    const progressBar = document.querySelector('.progress-bar');
    if (progressBar) {
        progressBar.style.width = '0%';
        progressBar.textContent = '0%';
    }
    
    // Limpiar textos de progreso
    const progressText = document.getElementById('progress-text');
    if (progressText) progressText.textContent = 'Esperando inicio...';
    
    const currentFrame = document.getElementById('current-frame');
    if (currentFrame) currentFrame.textContent = '0';
    
    const totalFrames = document.getElementById('total-frames');
    if (totalFrames) totalFrames.textContent = '0';
    
    console.log('Interface reset - ready for new analysis');
}

// File selection handler
document.addEventListener('DOMContentLoaded', function() {
    // Event listener para el logo
    const brandLogo = document.getElementById('brand-logo');
    if (brandLogo) {
        brandLogo.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Confirmar si hay un análisis en curso
            if (currentSessionId && websocket) {
                if (confirm('¿Estás seguro de que quieres cancelar el análisis actual y volver al inicio?')) {
                    resetInterface();
                }
            } else {
                resetInterface();
            }
        });
    }
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
        console.log('Stats recibidas:', data.stats);
        updateLiveStats(data.stats);
        updateLiveCharts(data.stats);
        
        // Actualizar estadísticas espaciales si están disponibles
        if (data.stats.spatial) {
            console.log('Spatial stats:', data.stats.spatial);
            updateSpatialStats(data.stats.spatial);
        } else {
            console.warn('No spatial stats en este batch');
        }
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
    console.log('Mostrando resultados finales:', stats);
    
    document.getElementById('progress-section').style.display = 'none';
    document.getElementById('results-section').style.display = 'block';
    
    // Stats Overview
    document.getElementById('total-time').textContent = stats.total_seconds.toFixed(1) + 's';
    document.getElementById('total-frames-stat').textContent = stats.total_frames;
    document.getElementById('possession-stat-0').textContent = (stats.possession_percent[0] || 0).toFixed(1) + '%';
    document.getElementById('possession-stat-1').textContent = (stats.possession_percent[1] || 0).toFixed(1) + '%';
    
    // Mostrar botón de resumen de heatmaps
    const summaryBtnContainer = document.getElementById('heatmap-summary-btn-container');
    if (summaryBtnContainer) {
        summaryBtnContainer.style.display = 'block';
    }
    
    // Actualizar estadísticas espaciales finales si están disponibles
    if (stats.spatial) {
        console.log('Stats espaciales finales:', stats.spatial);
        updateSpatialStats(stats.spatial);
    }
    
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
    
    // IMPORTANTE: Actualizar heatmaps cuando el análisis termina
    // Esperar un momento para que los archivos se guarden
    if (currentSessionId) {
        console.log('Programando carga de heatmaps finales...');
        setTimeout(() => {
            console.log('Cargando heatmaps finales...');
            updateHeatmapImages();
        }, 1000);
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

// Actualizar estadísticas espaciales
function updateSpatialStats(spatial) {
    console.log('updateSpatialStats llamada con:', spatial);
    
    if (!spatial) {
        console.warn('No hay datos espaciales');
        return;
    }
    
    // Mostrar sección de heatmaps
    const heatmapsSection = document.getElementById('spatial-heatmaps-section');
    if (heatmapsSection) {
        console.log('Mostrando sección de heatmaps');
        heatmapsSection.style.display = 'block';
    } else {
        console.error('No se encontró el elemento spatial-heatmaps-section');
    }
    
    // Actualizar estado de calibración
    const calibrationStatus = document.getElementById('calibration-status');
    const spatialStatusMessage = document.getElementById('spatial-status-message');
    
    if (calibrationStatus) {
        if (spatial.calibration_valid) {
            calibrationStatus.innerHTML = '<i class="fas fa-check-circle"></i> Calibrated';
            calibrationStatus.className = 'badge bg-success';
            if (spatialStatusMessage) {
                spatialStatusMessage.innerHTML = '<small>✅ Field calibration successful! Heatmaps are being generated.</small>';
            }
        } else {
            calibrationStatus.innerHTML = '<i class="fas fa-exclamation-triangle"></i> No Calibration';
            calibrationStatus.className = 'badge bg-warning';
            if (spatialStatusMessage) {
                spatialStatusMessage.innerHTML = '<small>⚠️ Field lines not detected. Heatmaps require visible field markings.</small>';
            }
        }
    }
    
    // Actualizar info de partición
    const partitionTypeText = document.getElementById('partition-type-text');
    if (partitionTypeText) {
        partitionTypeText.textContent = spatial.partition_type || 'thirds_lanes';
    }
    
    const numZonesText = document.getElementById('num-zones-text');
    if (numZonesText) {
        numZonesText.textContent = spatial.num_zones || 9;
    }
    
    // Actualizar heatmaps (usando sessionId actual)
    if (currentSessionId && spatial.calibration_valid) {
        updateHeatmapImages();
    }
    
    // Mostrar top zonas
    if (spatial.zone_percentages) {
        updateTopZones(0, spatial.zone_percentages['0'] || spatial.zone_percentages[0]);
        updateTopZones(1, spatial.zone_percentages['1'] || spatial.zone_percentages[1]);
    }
}

// Actualizar imágenes de heatmaps
function updateHeatmapImages() {
    if (!currentSessionId) {
        console.warn('No hay currentSessionId para actualizar heatmaps');
        return;
    }
    
    const timestamp = new Date().getTime();
    
    console.log('Actualizando heatmaps para session:', currentSessionId);
    
    const heatmapTeam0 = document.getElementById('heatmap-team-0');
    if (heatmapTeam0) {
        const url = `/api/heatmap/${currentSessionId}/0?t=${timestamp}`;
        console.log('Cargando heatmap Team 0:', url);
        heatmapTeam0.src = url;
    } else {
        console.error('No se encontró elemento heatmap-team-0');
    }
    
    const heatmapTeam1 = document.getElementById('heatmap-team-1');
    if (heatmapTeam1) {
        const url = `/api/heatmap/${currentSessionId}/1?t=${timestamp}`;
        console.log('Cargando heatmap Team 1:', url);
        heatmapTeam1.src = url;
    } else {
        console.error('No se encontró elemento heatmap-team-1');
    }
}

// Actualizar top zonas
function updateTopZones(teamId, zonePercentages) {
    if (!zonePercentages || !Array.isArray(zonePercentages)) return;
    
    const zoneNames = [
        'Defensive Left', 'Defensive Center', 'Defensive Right',
        'Midfield Left', 'Midfield Center', 'Midfield Right',
        'Offensive Left', 'Offensive Center', 'Offensive Right'
    ];
    
    // Crear array de zonas con índice y porcentaje
    const zones = zonePercentages.map((pct, idx) => ({
        index: idx,
        name: zoneNames[idx] || `Zone ${idx}`,
        percent: pct
    }));
    
    // Ordenar por porcentaje descendente
    zones.sort((a, b) => b.percent - a.percent);
    
    // Tomar top 3
    const top3 = zones.slice(0, 3).filter(z => z.percent > 0);
    
    // Actualizar HTML
    const topZonesDiv = document.getElementById(`top-zones-team-${teamId}`);
    if (topZonesDiv) {
        if (top3.length === 0) {
            topZonesDiv.innerHTML = '<span class="badge bg-secondary">No data yet</span>';
        } else {
            const badgeClass = teamId === 0 ? 'bg-success' : 'bg-danger';
            topZonesDiv.innerHTML = top3.map((zone, i) => 
                `<span class="badge ${badgeClass} me-1">${i + 1}. ${zone.name} (${zone.percent.toFixed(1)}%)</span>`
            ).join('');
        }
    }
}

// Función para mostrar el resumen de heatmaps
function showHeatmapSummary() {
    if (!currentSessionId) {
        console.error('No session ID available');
        return;
    }
    
    const summaryUrl = `/api/heatmap-summary/${currentSessionId}?t=${Date.now()}`;
    const summaryImg = document.getElementById('heatmap-summary-image');
    const downloadBtn = document.getElementById('download-summary-btn');
    
    // Actualizar imagen
    summaryImg.src = summaryUrl;
    
    // Actualizar botón de descarga
    downloadBtn.href = summaryUrl;
    
    // Mostrar modal
    const modal = new bootstrap.Modal(document.getElementById('heatmapSummaryModal'));
    modal.show();
}
