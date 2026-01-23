// Estado global
let currentSessionId = null;
let websocket = null;
let possessionChart = null;
let passesChart = null;
let timelineChart = null;

// File selection handler
document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('videoFile');
    const fileInfo = document.getElementById('file-info');
    const fileName = document.getElementById('file-name');
    const uploadZone = document.getElementById('uploadZone');
    
    fileInput.addEventListener('change', function(e) {
        if (this.files && this.files[0]) {
            fileName.textContent = this.files[0].name;
            fileInfo.style.display = 'block';
        }
    });
    
    // Drag and drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.style.borderColor = 'var(--accent-green)';
    });
    
    uploadZone.addEventListener('dragleave', () => {
        uploadZone.style.borderColor = 'var(--border-color)';
    });
    
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.style.borderColor = 'var(--border-color)';
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            fileName.textContent = files[0].name;
            fileInfo.style.display = 'block';
        }
    });
});

// Subir video
async function uploadVideo() {
    const fileInput = document.getElementById('videoFile');
    const file = fileInput.files[0];
    
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
        
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
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
    if (data.type === 'status') {
        document.getElementById('progress-text').textContent = data.message;
    } else if (data.type === 'progress') {
        updateProgress(data);
    } else if (data.type === 'completed') {
        showResults(data.stats);
    } else if (data.type === 'error') {
        showError(data.message);
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
    const elapsedTime = document.getElementById('elapsed-time');
    
    progressBarFill.style.width = data.progress + '%';
    progressPercent.textContent = data.progress + '%';
    
    if (data.frame !== undefined) {
        currentFrame.textContent = data.frame;
    }
    
    if (data.total_frames !== undefined) {
        totalFrames.textContent = data.total_frames;
    }
    
    if (data.elapsed_time !== undefined) {
        elapsedTime.textContent = Math.round(data.elapsed_time) + 's';
    }
    
    progressText.textContent = data.message || 'Processing video...';
    progressText.textContent = `Frame ${data.frame} / ${data.total_frames}`;
    
    // Actualizar estadísticas en tiempo real si existen
    if (data.stats) {
        updateLiveStats(data.stats);
    }
}

// Actualizar estadísticas en vivo
function updateLiveStats(stats) {
    // Mostrar sección de resultados si no está visible
    if (document.getElementById('results-section').style.display === 'none') {
        document.getElementById('results-section').style.display = 'block';
        initializeCharts();
    }
    
    // Actualizar posesión
    if (stats.possession_percent) {
        const p0 = stats.possession_percent[0] || 0;
        const p1 = stats.possession_percent[1] || 0;
        
        document.getElementById('possession-percent-0').textContent = p0.toFixed(1) + '%';
        document.getElementById('possession-percent-1').textContent = p1.toFixed(1) + '%';
        
        document.getElementById('possession-bar-0').style.width = p0 + '%';
        document.getElementById('possession-bar-1').style.width = p1 + '%';
    }
    
    // Actualizar tiempo
    if (stats.possession_seconds) {
        const t0 = stats.possession_seconds[0] || 0;
        const t1 = stats.possession_seconds[1] || 0;
        
        document.getElementById('possession-time-0').textContent = t0.toFixed(1) + 's';
        document.getElementById('possession-time-1').textContent = t1.toFixed(1) + 's';
    }
    
    // Actualizar gráficos
    if (possessionChart && stats.possession_percent) {
        possessionChart.data.datasets[0].data = [
            stats.possession_percent[0] || 0,
            stats.possession_percent[1] || 0
        ];
        possessionChart.update('none');
    }
    
    if (passesChart && stats.passes) {
        passesChart.data.datasets[0].data = [
            stats.passes[0] || 0,
            stats.passes[1] || 0
        ];
        passesChart.update('none');
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
