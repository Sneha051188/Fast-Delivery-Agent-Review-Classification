// filepath: c:\fast_delivery_ml_project\static\js\script.js

document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const reviewText = document.getElementById('reviewText');
    const predictBtn = document.getElementById('predictBtn');
    const resultCard = document.getElementById('resultCard');
    const resultHeader = document.getElementById('resultHeader');
    const resultTitle = document.getElementById('resultTitle');
    const predictionLabel = document.getElementById('predictionLabel');
    const predictionModel = document.getElementById('predictionModel');
    const confidenceBar = document.getElementById('confidenceBar');
    const correctBtn = document.getElementById('correctBtn');
    const incorrectBtn = document.getElementById('incorrectBtn');
    const exampleItems = document.querySelectorAll('.example-item');
    const agentsTab = document.getElementById('agents-tab');

    let mapInitialized = false;
    let currentPrediction = {};

    /* ----------------------------- Utility Functions ----------------------------- */

    const showToast = (message, type = 'info') => {
        const toast = document.createElement('div');
        toast.className = `toast align-items-center text-white border-0 position-fixed top-0 end-0 m-3 bg-${type}`;
        toast.role = 'alert';
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">${message}</div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        `;
        document.body.appendChild(toast);
        const bsToast = new bootstrap.Toast(toast, { delay: 3000 });
        bsToast.show();
        toast.addEventListener('hidden.bs.toast', () => toast.remove());
    };

    const fetchJSON = async (url, method = 'GET', data = null) => {
        const options = {
            method,
            headers: { 'Content-Type': 'application/json' },
        };
        if (data) options.body = JSON.stringify(data);

        const response = await fetch(url, options);
        if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        return await response.json();
    };

    /* ----------------------------- Example Click Handler ----------------------------- */

    exampleItems.forEach(item => {
        item.addEventListener('click', () => {
            reviewText.value = item.getAttribute('data-text');
            reviewText.focus();
            reviewText.scrollIntoView({ behavior: 'smooth', block: 'center' });
        });
    });

    /* ----------------------------- Prediction Handler ----------------------------- */

    predictBtn.addEventListener('click', async () => {
        const text = reviewText.value.trim();
        if (!text) return showToast('Please enter a review text!', 'warning');

        const model = document.querySelector('input[name="modelChoice"]:checked')?.value || 'tfidf';
        predictBtn.disabled = true;
        predictBtn.innerHTML = `<span class="spinner-border spinner-border-sm me-2"></span>Predicting...`;

        try {
            const result = await fetchJSON('/api/predict', 'POST', { text, model });

            currentPrediction = { text, label: result.label, confidence: result.confidence, model };

            updatePredictionUI(result);
            showToast('Prediction completed successfully!', 'success');
        } catch (error) {
            console.error('Prediction Error:', error);
            showToast(`Error: ${error.message}`, 'danger');
        } finally {
            predictBtn.disabled = false;
            predictBtn.innerHTML = `<i class="fas fa-magic me-2"></i>Predict Sentiment`;
        }
    });

    /* ----------------------------- UI Update Logic ----------------------------- */

    function updatePredictionUI(result) {
        resultCard.classList.remove('d-none');

        const label = result.label?.toLowerCase() || '';
        const isPositive = ['positive', 'correct', 'good'].some(k => label.includes(k));

        if (isPositive) {
            resultHeader.className = 'card-header bg-success text-white';
            resultTitle.textContent = 'Prediction: Correct Delivery';
            predictionLabel.textContent = 'Correct Delivery ✓';
            predictionLabel.className = 'text-success fw-bold';
        } else {
            resultHeader.className = 'card-header bg-danger text-white';
            resultTitle.textContent = 'Prediction: Incorrect Delivery';
            predictionLabel.textContent = 'Incorrect Delivery ✗';
            predictionLabel.className = 'text-danger fw-bold';
        }

        predictionModel.textContent = `Model: ${result.model === 'tfidf' ? 'TF-IDF' : 'DistilBERT'}`;
        const confidence = Math.round((result.confidence || 0) * 100);
        confidenceBar.style.width = `${confidence}%`;
        confidenceBar.textContent = `${confidence}%`;
        confidenceBar.className = `progress-bar ${confidence < 50 ? 'bg-danger' : confidence < 75 ? 'bg-warning' : 'bg-success'}`;

        resultCard.scrollIntoView({ behavior: 'smooth' });
    }

    /* ----------------------------- Feedback Handler ----------------------------- */

    const handleFeedback = async (userLabel) => {
        if (!currentPrediction.text) return;

        try {
            await fetchJSON('/api/feedback', 'POST', {
                text: currentPrediction.text,
                predicted_label: currentPrediction.label,
                confidence: currentPrediction.confidence,
                model: currentPrediction.model,
                user_label: userLabel
            });

            showToast('Thank you for your feedback!', 'success');
            reviewText.value = '';
            resultCard.classList.add('d-none');
            currentPrediction = {};
        } catch (error) {
            console.error('Feedback Error:', error);
            showToast('Error submitting feedback.', 'danger');
        }
    };

    correctBtn.addEventListener('click', () => handleFeedback('Correct'));
    incorrectBtn.addEventListener('click', () => handleFeedback('Incorrect'));

    /* ----------------------------- Chart Metrics Loader ----------------------------- */

    const loadMetricsChart = async () => {
        try {
            const metrics = await fetchJSON('/api/metrics');
            const ctx = document.getElementById('metricsChart').getContext('2d');

            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['Accuracy', 'F1 Score'],
                    datasets: [
                        {
                            label: 'TF-IDF',
                            data: [metrics.tfidf.accuracy, metrics.tfidf.f1],
                            backgroundColor: 'rgba(13, 110, 253, 0.5)',
                            borderColor: '#0d6efd',
                            borderWidth: 1
                        },
                        {
                            label: 'DistilBERT',
                            data: [metrics.distilbert.accuracy, metrics.distilbert.f1],
                            backgroundColor: 'rgba(220, 53, 69, 0.5)',
                            borderColor: '#dc3545',
                            borderWidth: 1
                        }
                    ]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 1,
                            title: { display: true, text: 'Score' }
                        }
                    },
                    plugins: {
                        legend: { position: 'top' },
                        title: { display: true, text: 'Model Performance Comparison' }
                    }
                }
            });
        } catch (error) {
            console.error('Chart Error:', error);
            showToast('Error loading model metrics.', 'danger');
        }
    };

    /* ----------------------------- Map Loader ----------------------------- */

    const initMap = () => {
        const agents = [
            { name: "Raj", lat: 28.6139, lng: 77.2090, rating: 4.8 },
            { name: "Aisha", lat: 28.7041, lng: 77.1025, rating: 4.5 },
            { name: "Karan", lat: 28.4595, lng: 77.0266, rating: 4.9 },
            { name: "Sneha", lat: 28.5355, lng: 77.3910, rating: 4.6 },
            { name: "Vikram", lat: 28.4089, lng: 77.3178, rating: 4.4 }
        ];

        const map = L.map('map').setView([28.6139, 77.2090], 11);

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(map);

        agents.forEach(agent => {
            const marker = L.marker([agent.lat, agent.lng]).addTo(map);
            marker.bindPopup(`<b>${agent.name}</b><br>Rating: ${agent.rating} ⭐`);
        });
    };

    agentsTab.addEventListener('shown.bs.tab', () => {
        if (!mapInitialized) {
            initMap();
            mapInitialized = true;
        }
    });

    /* ----------------------------- Initialize Dashboard ----------------------------- */
    loadMetricsChart();
});




