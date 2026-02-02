function analyzeNews() {
    const text = document.getElementById('newsInput').value;
    if (!text.trim()) {
        alert("Please enter some text to analyze!");
        return;
    }

    // UI Loading State
    const btn = document.getElementById('analyzeBtn');
    const btnText = document.getElementById('btnText');
    const loader = document.getElementById('loader');
    const resultSection = document.getElementById('resultSection');

    btn.disabled = true;
    btnText.style.opacity = '0';
    loader.style.display = 'block';
    resultSection.classList.add('hidden');

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: text }),
    })
        .then(response => response.json())
        .then(data => {
            // Reset Button
            btn.disabled = false;
            btnText.style.opacity = '1';
            loader.style.display = 'none';

            // Show Results
            displayResults(data);
        })
        .catch((error) => {
            console.error('Error:', error);
            btn.disabled = false;
            btnText.style.opacity = '1';
            loader.style.display = 'none';
            alert("An error occurred. Please try again.");
        });
}

function displayResults(data) {
    const resultSection = document.getElementById('resultSection');
    const veracityBadge = document.getElementById('veracityBadge');
    const meterFill = document.getElementById('meterFill');
    const confidenceScore = document.getElementById('confidenceScore');
    const keywordsList = document.getElementById('keywordsList');

    resultSection.classList.remove('hidden');

    // Update Badge
    veracityBadge.textContent = data.prediction;
    veracityBadge.className = 'badge ' + data.prediction.toLowerCase(); // 'real' or 'fake'

    // Update Confidence Meter
    // Format "95.5%" -> 95.5
    const percentage = parseFloat(data.confidence);
    meterFill.style.width = percentage + '%';

    // Set color based on prediction
    if (data.prediction === 'REAL') {
        meterFill.style.backgroundColor = 'var(--accent-real)';
    } else {
        meterFill.style.backgroundColor = 'var(--accent-fake)';
    }

    confidenceScore.textContent = data.confidence + ' Confidence';

    // Update Keywords
    keywordsList.innerHTML = '';
    data.explanation.forEach(item => {
        // item is [word, weight]
        const word = item[0];
        const weight = item[1];

        const tag = document.createElement('div');
        // If weight > 0 it typically means it pushes towards the positive class (Real), else Fake
        // This depends on the model. Assuming 1=REAL, 0=FAKE.
        const type = weight > 0 ? 'positive' : 'negative';

        tag.className = `keyword-tag ${type}`;
        tag.innerHTML = `<span>${word}</span> <small>(${weight.toFixed(3)})</small>`;
        keywordsList.appendChild(tag);
    });
}
