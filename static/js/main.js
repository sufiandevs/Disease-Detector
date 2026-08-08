function switchTab(method) {
    document.getElementById('inputMethod').value = method;
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    event.target.closest('.tab-btn').classList.add('active');
    document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.remove('active'));
    document.getElementById(method + 'Panel').classList.add('active');
}

function filterSymptoms() {
    const search = document.getElementById('symptomSearch').value.toLowerCase();
    document.querySelectorAll('.symptom-chip').forEach(chip => {
        chip.style.display = chip.textContent.toLowerCase().includes(search) ? 'block' : 'none';
    });
}

document.addEventListener('change', function(e) {
    if (e.target.name === 'symptoms') {
        document.getElementById('selectedCount').textContent = 
            document.querySelectorAll('input[name="symptoms"]:checked').length;
    }
});

function addUserMessage() {
    const chatText = document.getElementById('chatText').value.trim();
    const chatMessages = document.getElementById('chatMessages');
    if (!chatText) return;
    
    const userMsg = document.createElement('div');
    userMsg.className = 'message user';
    userMsg.style.justifyContent = 'flex-end';
    userMsg.innerHTML = `<div class="message-bubble" style="background: linear-gradient(135deg, #0ea5e9, #6366f1); color: white; border-bottom-right-radius: 4px; border-bottom-left-radius: 18px;">${chatText}</div>`;
    chatMessages.appendChild(userMsg);
    
    setTimeout(() => {
        const botMsg = document.createElement('div');
        botMsg.className = 'message bot';
        botMsg.innerHTML = `<div class="message-avatar">🤖</div><div class="message-bubble">Thank you for describing your symptoms. Click <strong>"Analyze My Symptoms"</strong> below to get your results.</div>`;
        chatMessages.appendChild(botMsg);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }, 600);
    
    document.getElementById('chatText').value = '';
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

const quotes = [
    "Analyzing symptom patterns across 200+ medical conditions...",
    "Cross-referencing with advanced medical databases...",
    "Evaluating probabilistic disease models...",
    "Consulting AI diagnostic neural networks...",
    "Generating personalized health insights...",
    "Comparing against clinical symptom profiles...",
    "Finalizing diagnostic possibilities..."
];

function showLoading() {
    const method = document.getElementById('inputMethod').value;
    let valid = false;
    if (method === 'checkbox') {
        valid = document.querySelectorAll('input[name="symptoms"]:checked').length > 0;
    } else {
        valid = document.getElementById('chatText').value.trim().length > 0;
    }
    
    if (!valid) {
        alert('Please select or describe at least one symptom before analyzing.');
        return false;
    }
    
    const overlay = document.getElementById('loadingOverlay');
    const quoteEl = document.getElementById('loadingQuote');
    overlay.classList.remove('hidden');
    
    let quoteIndex = 0;
    quoteEl.textContent = quotes[0];
    setInterval(() => {
        quoteIndex = (quoteIndex + 1) % quotes.length;
        quoteEl.style.opacity = '0';
        setTimeout(() => { quoteEl.textContent = quotes[quoteIndex]; quoteEl.style.opacity = '1'; }, 500);
    }, 2500);
    
    return true;
}
