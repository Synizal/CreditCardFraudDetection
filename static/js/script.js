document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const transactionForm = document.getElementById('transactionForm');
    const modelSelect = document.getElementById('modelSelect');
    const modelSelectors = document.querySelectorAll('.model-selector');
    const randomFeaturesBtn = document.getElementById('randomFeaturesBtn');
    const featureInputs = document.querySelectorAll('.feature-input');
    const resultsCard = document.getElementById('resultsCard');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const fraudAlert = document.getElementById('fraudAlert');
    const resultAmount = document.getElementById('resultAmount');
    const resultTime = document.getElementById('resultTime');
    const resultModel = document.getElementById('resultModel');
    const resultProbability = document.getElementById('resultProbability');

    // Current selected model
    let selectedModel = '';

    // Set up event listeners
    if (transactionForm) {
        transactionForm.addEventListener('submit', submitTransaction);
    }

    if (modelSelect) {
        modelSelect.addEventListener('change', function() {
            selectedModel = this.value;
            // Deselect top model buttons when dropdown changes
            modelSelectors.forEach(btn => btn.classList.remove('active'));
        });
    }

    // Top model selection buttons
    modelSelectors.forEach(button => {
        button.addEventListener('click', function() {
            // Remove active class from all buttons
            modelSelectors.forEach(btn => btn.classList.remove('active'));
            // Add active class to clicked button
            this.classList.add('active');
            // Set selected model
            selectedModel = this.dataset.modelId;
            // Reset the dropdown selection
            if (modelSelect) {
                modelSelect.value = '';
            }
        });
    });

    // Generate random features button
    if (randomFeaturesBtn) {
        randomFeaturesBtn.addEventListener('click', generateRandomFeatures);
    }

    // Submit transaction form
    function submitTransaction(e) {
        e.preventDefault();
        
        // Show loading state
        resultsCard.style.display = 'block';
        loadingSpinner.style.display = 'block';
        document.getElementById('resultsContent').style.display = 'none';
        
        // Gather form data
        const formData = new FormData(transactionForm);
        const transaction = {};
        
        // Convert form data to transaction object
        for (const [key, value] of formData.entries()) {
            if (value !== '') {
                transaction[key] = parseFloat(value);
            }
        }
        
        // Prepare request payload
        const payload = {
            transaction: transaction
        };
        
        // Add model name if one is selected
        if (selectedModel) {
            payload.model_name = selectedModel;
        }
        
        // Make API request
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        })
        .then(response => response.json())
        .then(data => {
            displayResults(data);
        })
        .catch(error => {
            console.error('Lỗi:', error);
            displayError('Đã xảy ra lỗi khi xử lý yêu cầu của bạn.');
        })
        .finally(() => {
            loadingSpinner.style.display = 'none';
            document.getElementById('resultsContent').style.display = 'block';
        });
    }

    // Display prediction results
    function displayResults(data) {
        if (data.error) {
            displayError(data.error);
            return;
        }
        
        const prediction = data.predictions[0];
        
        // Update result elements
        resultAmount.textContent = `$${prediction.amount.toFixed(2)}`;
        resultTime.textContent = secondsToTime(prediction.time);
        resultModel.textContent = formatModelName(data.model_used);
        resultProbability.textContent = `${(prediction.fraud_probability * 100).toFixed(2)}%`;
        
        // Update fraud alert
        if (prediction.is_fraud) {
            fraudAlert.className = 'alert alert-danger';
            fraudAlert.innerHTML = '<i class="bi bi-exclamation-triangle-fill"></i> Phát hiện giao dịch gian lận!';
        } else {
            fraudAlert.className = 'alert alert-success';
            fraudAlert.innerHTML = '<i class="bi bi-shield-check"></i> Giao dịch hợp lệ';
        }
    }

    // Display error message
    function displayError(message) {
        fraudAlert.className = 'alert alert-warning';
        fraudAlert.textContent = message;
        
        // Clear other result fields
        resultAmount.textContent = '';
        resultTime.textContent = '';
        resultModel.textContent = '';
        resultProbability.textContent = '';
    }

    // Generate random feature values
    function generateRandomFeatures() {
        featureInputs.forEach(input => {
            // Generate random values between -5 and 5 with 6 decimal places
            const randomValue = (Math.random() * 10 - 5).toFixed(6);
            input.value = randomValue;
        });
    }

    // Format seconds to readable time
    function secondsToTime(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        
        if (hours > 0) {
            return `${hours}h ${minutes}p ${secs}s`;
        } else if (minutes > 0) {
            return `${minutes}p ${secs}s`;
        } else {
            return `${secs}s`;
        }
    }

    // Format model name for display
    function formatModelName(modelName) {
        if (!modelName) return 'Không xác định';
        
        // Replace underscores with spaces and capitalize words
        return modelName
            .replace(/_/g, ' ')
            .replace(/smote/i, '(SMOTE)')
            .replace(/\w\S*/g, function(txt) {
                return txt.charAt(0).toUpperCase() + txt.substr(1).toLowerCase();
            });
    }
}); 