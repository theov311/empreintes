// frontend/assets/js/app.js

// API Configuration
const API_BASE_URL = 'http://localhost:5000/api';
const API_ENDPOINTS = {
    identify: `${API_BASE_URL}/identify`,
    species: `${API_BASE_URL}/species`,
    observations: `${API_BASE_URL}/observations`,
    stats: `${API_BASE_URL}/stats`,
    status: `${API_BASE_URL}/status`,
    uploads: `${API_BASE_URL}/uploads`
};

// App state
const appState = {
    cameraActive: false,
    photoTaken: false,
    processingRequest: false,
    currentStream: null,
    currentLocation: null,
    identificationResult: null,
    observations: [],
    species: [],
    stats: null
};

// DOM Elements
const elements = {
    // Camera elements
    cameraContainer: document.getElementById('cameraContainer'),
    camera: document.getElementById('camera'),
    canvas: document.getElementById('canvas'),
    photo: document.getElementById('photo'),
    captureBtn: document.getElementById('captureBtn'),
    retakeBtn: document.getElementById('retakeBtn'),
    
    // Result elements
    loader: document.getElementById('loader'),
    resultCard: document.getElementById('resultCard'),
    animalName: document.getElementById('animalName'),
    confidenceValue: document.getElementById('confidenceValue'),
    confidenceBar: document.getElementById('confidenceBar'),
    habitat: document.getElementById('habitat'),
    behavior: document.getElementById('behavior'),
    size: document.getElementById('size'),
    diet: document.getElementById('diet'),
    status: document.getElementById('status'),
    description: document.getElementById('description'),
    alternatives: document.getElementById('alternatives'),
    
    // Observations elements
    observationsGrid: document.getElementById('observationsGrid'),
    
    // Stats elements
    totalObservations: document.getElementById('totalObservations'),
    statsContainer: document.getElementById('statsContainer'),
    speciesChart: document.getElementById('speciesChart'),
    dateChart: document.getElementById('dateChart')
};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    // Initialize any page-specific functionality
    initCurrentPage();
    
    // Check API status
    checkApiStatus();
});

// Initialize functionality based on current page
function initCurrentPage() {
    const currentPath = window.location.pathname;
    
    // Home page or identify page
    if (currentPath === '/' || currentPath === '/index.html' || currentPath === '/identify.html') {
        initCameraPage();
    } 
    // Observations page
    else if (currentPath === '/observations.html') {
        loadObservations();
    } 
    // Stats page
    else if (currentPath === '/stats.html') {
        loadStats();
    }
}

// Check API status
async function checkApiStatus() {
    try {
        const response = await fetch(API_ENDPOINTS.status);
        if (!response.ok) {
            throw new Error('API is not available');
        }
        
        const data = await response.json();
        console.log('API Status:', data);
        
        if (!data.model_loaded) {
            showAlert('Le modèle n\'est pas chargé. Certaines fonctionnalités peuvent ne pas fonctionner correctement.', 'warning');
        }
    } catch (error) {
        console.error('Error checking API status:', error);
        showAlert('L\'API n\'est pas accessible. Veuillez vérifier que le serveur est en cours d\'exécution.', 'error');
    }
}

// Initialize camera page
function initCameraPage() {
    if (!elements.cameraContainer) return;
    
    // Get user's location if possible
    getLocation();
    
    // Initialize camera
    startCamera();
    
    // Add event listeners
    if (elements.captureBtn) {
        elements.captureBtn.addEventListener('click', capturePhoto);
    }
    
    if (elements.retakeBtn) {
        elements.retakeBtn.addEventListener('click', retakePhoto);
    }
}

// Get user's location
function getLocation() {
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
            position => {
                appState.currentLocation = {
                    latitude: position.coords.latitude,
                    longitude: position.coords.longitude
                };
                console.log('Location acquired:', appState.currentLocation);
            },
            error => {
                console.warn('Error getting location:', error);
                appState.currentLocation = null;
            }
        );
    } else {
        console.warn('Geolocation is not supported by this browser.');
        appState.currentLocation = null;
    }
}

// Start camera
async function startCamera() {
    if (!elements.camera) return;
    
    if (appState.currentStream) {
        // Stop any existing stream
        appState.currentStream.getTracks().forEach(track => track.stop());
    }
    
    try {
        // Request camera access with environment facing camera if on mobile
        const constraints = { 
            video: { 
                facingMode: isMobile() ? 'environment' : 'user',
                width: { ideal: 1280 },
                height: { ideal: 720 }
            }, 
            audio: false 
        };
        
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        elements.camera.srcObject = stream;
        appState.currentStream = stream;
        appState.cameraActive = true;
        
        // Show camera view, hide photo
        elements.camera.style.display = 'block';
        elements.photo.style.display = 'none';
        elements.captureBtn.style.display = 'inline-block';
        elements.retakeBtn.style.display = 'none';
        
    } catch (err) {
        console.error('Error accessing camera:', err);
        showAlert('Impossible d\'accéder à la caméra. Veuillez vérifier les permissions.', 'error');
    }
}

// Capture photo
function capturePhoto() {
    if (!elements.camera || !elements.canvas || !elements.photo) return;
    
    // Set canvas dimensions to match video
    elements.canvas.width = elements.camera.videoWidth;
    elements.canvas.height = elements.camera.videoHeight;
    
    // Draw video frame to canvas
    elements.canvas.getContext('2d').drawImage(elements.camera, 0, 0, elements.canvas.width, elements.canvas.height);
    
    // Convert to image
    const dataUrl = elements.canvas.toDataURL('image/jpeg', 0.8);
    
    // Display captured image
    elements.photo.src = dataUrl;
    elements.camera.style.display = 'none';
    elements.photo.style.display = 'block';
    elements.captureBtn.style.display = 'none';
    elements.retakeBtn.style.display = 'inline-block';
    
    appState.photoTaken = true;
    
    // Show loading indicator
    if (elements.loader) {
        elements.loader.style.display = 'block';
    }
    
    // Hide any previous result
    if (elements.resultCard) {
        elements.resultCard.style.display = 'none';
    }
    
    // Send image to API
    identifyFootprint(dataUrl);
}

// Retake photo
function retakePhoto() {
    if (!elements.camera || !elements.photo) return;
    
    // Reset UI
    elements.photo.style.display = 'none';
    elements.camera.style.display = 'block';
    elements.captureBtn.style.display = 'inline-block';
    elements.retakeBtn.style.display = 'none';
    
    // Reset state
    appState.photoTaken = false;
    
    // Hide result
    if (elements.resultCard) {
        elements.resultCard.style.display = 'none';
    }
    
    // Hide loader
    if (elements.loader) {
        elements.loader.style.display = 'none';
    }
}

// Send image to API for identification
async function identifyFootprint(imageData) {
    if (appState.processingRequest) return;
    
    appState.processingRequest = true;
    
    try {
        const response = await fetch(API_ENDPOINTS.identify, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: imageData,
                latitude: appState.currentLocation?.latitude || null,
                longitude: appState.currentLocation?.longitude || null
            })
        });
        
        if (!response.ok) {
            throw new Error(`Error: ${response.status} ${response.statusText}`);
        }
        
        const data = await response.json();
        console.log('Identification result:', data);
        
        // Store result in app state
        appState.identificationResult = data;
        
        // Display result
        displayIdentificationResult(data);
        
    } catch (error) {
        console.error('Error identifying footprint:', error);
        showAlert('Erreur lors de l\'identification. Veuillez réessayer.', 'error');
        
        // Hide loader
        if (elements.loader) {
            elements.loader.style.display = 'none';
        }
    } finally {
        appState.processingRequest = false;
    }
}

// Display identification result
function displayIdentificationResult(data) {
    // Hide loader
    if (elements.loader) {
        elements.loader.style.display = 'none';
    }
    
    // Check if we have all the required elements
    if (!elements.resultCard || !elements.animalName) return;
    
    // Set animal name
    elements.animalName.textContent = data.species;
    
    // Set confidence
    if (elements.confidenceValue) {
        const confidencePercent = Math.round(data.confidence * 100);
        elements.confidenceValue.textContent = `${confidencePercent}%`;
        
        // Update confidence bar if it exists
        if (elements.confidenceBar) {
            elements.confidenceBar.style.width = `${confidencePercent}%`;
            
            // Change color based on confidence
            if (confidencePercent < 50) {
                elements.confidenceBar.style.backgroundColor = '#dc3545'; // Danger/red
            } else if (confidencePercent < 75) {
                elements.confidenceBar.style.backgroundColor = '#ffc107'; // Warning/yellow
            } else {
                elements.confidenceBar.style.backgroundColor = '#28a745'; // Success/green
            }
        }
    }
    
    // Set animal information
    if (data.info) {
        if (elements.habitat) elements.habitat.textContent = data.info.habitat || '-';
        if (elements.behavior) elements.behavior.textContent = data.info.comportement || '-';
        if (elements.size) elements.size.textContent = data.info.taille || '-';
        if (elements.diet) elements.diet.textContent = data.info.alimentation || '-';
        if (elements.status) elements.status.textContent = data.info.statut || '-';
        if (elements.description) elements.description.textContent = data.info.description || '-';
    }
    
    // Set alternatives
    if (elements.alternatives && data.top3 && data.top3.length > 1) {
        elements.alternatives.innerHTML = '';
        
        data.top3.slice(1).forEach(alt => {
            const item = document.createElement('li');
            item.className = 'list-group-item d-flex justify-content-between align-items-center';
            
            const confidencePercent = Math.round(alt.confidence * 100);
            item.innerHTML = `
                ${alt.species}
                <span class="badge bg-primary rounded-pill">${confidencePercent}%</span>
            `;
            
            elements.alternatives.appendChild(item);
        });
    } else if (elements.alternatives) {
        elements.alternatives.innerHTML = '<li class="list-group-item">Aucune autre correspondance significative</li>';
    }
    
    // Show result card
    elements.resultCard.style.display = 'block';
}

// Load observations
async function loadObservations() {
    if (!elements.observationsGrid) return;
    
    try {
        // Show loading state
        elements.observationsGrid.innerHTML = '<div class="loader"></div>';
        
        // Fetch observations
        const response = await fetch(API_ENDPOINTS.observations);
        if (!response.ok) {
            throw new Error(`Error: ${response.status} ${response.statusText}`);
        }
        
        const data = await response.json();
        console.log('Observations:', data);
        
        // Store in app state
        appState.observations = data;
        
        // Display observations
        displayObservations(data);
        
    } catch (error) {
        console.error('Error loading observations:', error);
        elements.observationsGrid.innerHTML = `
            <div class="alert alert-danger">
                Erreur lors du chargement des observations: ${error.message}
            </div>
        `;
    }
}

// Display observations
function displayObservations(observations) {
    if (!elements.observationsGrid) return;
    
    // Clear previous content
    elements.observationsGrid.innerHTML = '';
    
    if (observations.length === 0) {
        elements.observationsGrid.innerHTML = `
            <div class="alert alert-info">
                Aucune observation n'a été enregistrée.
            </div>
        `;
        return;
    }
    
    // Create cards for each observation
    observations.forEach(obs => {
        const card = document.createElement('div');
        card.className = 'observation-card';
        
        const confidencePercent = Math.round(obs.confidence * 100);
        
        card.innerHTML = `
            <div class="observation-image">
                <img src="${obs.image_url || '../assets/img/placeholder.jpg'}" alt="${obs.species}">
            </div>
            <div class="observation-details">
                <h3 class="observation-species">${obs.species}</h3>
                <p class="observation-date">${formatDate(obs.date)} ${obs.time}</p>
                <p class="observation-location">
                    ${obs.latitude && obs.longitude ? `<i class="fas fa-map-marker-alt"></i> ${obs.latitude.toFixed(6)}, ${obs.longitude.toFixed(6)}` : 'Localisation non disponible'}
                </p>
                <div class="confidence-bar mt-2">
                    <div class="confidence-level" style="width: ${confidencePercent}%;"></div>
                </div>
                <p class="mt-1 text-right">${confidencePercent}% de confiance</p>
            </div>
        `;
        
        elements.observationsGrid.appendChild(card);
    });
}

// Load stats
async function loadStats() {
    if (!elements.statsContainer) return;
    
    try {
        // Show loading state
        elements.statsContainer.innerHTML = '<div class="loader"></div>';
        
        // Fetch stats
        const response = await fetch(API_ENDPOINTS.stats);
        if (!response.ok) {
            throw new Error(`Error: ${response.status} ${response.statusText}`);
        }
        
        const data = await response.json();
        console.log('Stats:', data);
        
        // Store in app state
        appState.stats = data;
        
        // Display stats
        displayStats(data);
        
    } catch (error) {
        console.error('Error loading stats:', error);
        elements.statsContainer.innerHTML = `
            <div class="alert alert-danger">
                Erreur lors du chargement des statistiques: ${error.message}
            </div>
        `;
    }
}

// Display stats
function displayStats(stats) {
    if (!elements.statsContainer) return;
    
    // Display total observations
    if (elements.totalObservations) {
        elements.totalObservations.textContent = stats.total_observations || 0;
    }
    
    // Initialize charts if the elements exist
    if (elements.speciesChart && stats.by_species) {
        createSpeciesChart(stats.by_species);
    }
    
    if (elements.dateChart && stats.by_date) {
        createDateChart(stats.by_date);
    }
}

// Create species chart
function createSpeciesChart(speciesData) {
    if (!elements.speciesChart) return;
    
    // Sort data by count in descending order
    speciesData.sort((a, b) => b.count - a.count);
    
    // Prepare data for chart
    const labels = speciesData.map(item => item.species);
    const data = speciesData.map(item => item.count);
    
    // Create chart
    const ctx = elements.speciesChart.getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Nombre d\'observations',
                data: data,
                backgroundColor: 'rgba(44, 119, 68, 0.7)',
                borderColor: 'rgba(44, 119, 68, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Nombre d\'observations'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Espèces'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Observations par espèce'
                }
            }
        }
    });
}

// Create date chart
function createDateChart(dateData) {
    if (!elements.dateChart) return;
    
    // Sort data by date
    dateData.sort((a, b) => new Date(a.date) - new Date(b.date));
    
    // Prepare data for chart
    const labels = dateData.map(item => formatDate(item.date));
    const data = dateData.map(item => item.count);
    
    // Create chart
    const ctx = elements.dateChart.getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Observations',
                data: data,
                fill: false,
                backgroundColor: 'rgba(245, 166, 35, 0.7)',
                borderColor: 'rgba(245, 166, 35, 1)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Nombre d\'observations'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Date'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Évolution des observations'
                }
            }
        }
    });
}

// Utility functions
function formatDate(dateStr) {
    if (!dateStr) return '';
    
    const date = new Date(dateStr);
    return new Intl.DateTimeFormat('fr-FR', { 
        day: '2-digit', 
        month: '2-digit', 
        year: 'numeric' 
    }).format(date);
}

function isMobile() {
    return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
}

function showAlert(message, type = 'info') {
    // Create alert element
    const alertElement = document.createElement('div');
    alertElement.className = `alert alert-${type} alert-dismissible fade show`;
    alertElement.role = 'alert';
    alertElement.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    // Add to document
    document.body.prepend(alertElement);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        alertElement.classList.remove('show');
        setTimeout(() => alertElement.remove(), 150);
    }, 5000);
}