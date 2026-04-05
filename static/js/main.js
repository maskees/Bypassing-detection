/**
 * Adversarial Defense Analysis — Frontend JavaScript
 * Handles: Attack Lab interactions, Chart.js visualizations,
 *          AJAX API calls, and Comparison Dashboard rendering.
 */

// ─── State ───
const state = {
    selectedImage: null,
    selectedImageIndex: 0,
    attackRunning: false,
    samples: [],
};

// ─── Utility ───
function $(sel) { return document.querySelector(sel); }
function $$(sel) { return document.querySelectorAll(sel); }

function showLoading(container, message = 'Processing...') {
    const overlay = document.createElement('div');
    overlay.className = 'loading-overlay';
    overlay.id = 'loading-overlay';
    overlay.innerHTML = `<div class="spinner"></div><div class="loading-text">${message}</div>`;
    container.style.position = 'relative';
    container.appendChild(overlay);
}

function hideLoading(container) {
    const overlay = container.querySelector('#loading-overlay');
    if (overlay) overlay.remove();
}

// ─── Attack Lab ───

async function loadSampleImages() {
    const gallery = $('#image-gallery');
    if (!gallery) return;

    try {
        const res = await fetch('/api/images?count=10');
        const data = await res.json();
        state.samples = data.samples;

        gallery.innerHTML = '';
        data.samples.forEach((sample, i) => {
            const thumb = document.createElement('div');
            thumb.className = `image-thumb ${i === 0 ? 'selected' : ''}`;
            thumb.onclick = () => selectImage(i);
            thumb.innerHTML = `
                <img src="data:image/png;base64,${sample.image}" alt="Digit ${sample.label}">
                <span class="thumb-label">${sample.label}</span>
            `;
            gallery.appendChild(thumb);
        });

        if (data.samples.length > 0) {
            selectImage(0);
        }
    } catch (err) {
        console.error('Failed to load samples:', err);
        gallery.innerHTML = '<div class="empty-state"><p>Failed to load images</p></div>';
    }
}

function selectImage(index) {
    state.selectedImageIndex = index;
    state.selectedImage = state.samples[index];

    $$('.image-thumb').forEach((el, i) => {
        el.classList.toggle('selected', i === index);
    });

    // Show original image preview
    const origContainer = $('#orig-image');
    if (origContainer) {
        origContainer.innerHTML = `<img src="data:image/png;base64,${state.selectedImage.image}" alt="Original">`;
    }

    const origLabel = $('#orig-label');
    if (origLabel) {
        origLabel.textContent = `True Label: ${state.selectedImage.label}`;
    }

    // Clear previous results
    clearResults();
}

function clearResults() {
    const advContainer = $('#adv-image');
    const pertContainer = $('#pert-image');
    if (advContainer) advContainer.innerHTML = '<div class="empty-state"><div class="empty-icon">🎯</div><p>Run an attack to see results</p></div>';
    if (pertContainer) pertContainer.innerHTML = '<div class="empty-state"><div class="empty-icon">🔍</div><p>Perturbation will appear here</p></div>';

    const metricsRow = $('#attack-metrics');
    if (metricsRow) metricsRow.innerHTML = '';

    const defenseGrid = $('#defense-results');
    if (defenseGrid) defenseGrid.innerHTML = '';

    // Clear confidence bars
    ['orig', 'adv'].forEach(prefix => {
        for (let i = 0; i < 10; i++) {
            const fill = $(`#${prefix}-bar-${i}`);
            const val = $(`#${prefix}-val-${i}`);
            if (fill) fill.style.width = '0%';
            if (val) val.textContent = '';
        }
    });
}

function updateEpsilonDisplay() {
    const slider = $('#epsilon-slider');
    const display = $('#epsilon-value');
    if (slider && display) {
        display.textContent = parseFloat(slider.value).toFixed(2);
    }
}

async function runAttack() {
    if (state.attackRunning || !state.selectedImage) return;
    state.attackRunning = true;

    const btn = $('#run-attack-btn');
    const resultsPanel = $('#results-panel');
    btn.disabled = true;
    btn.innerHTML = '<div class="spinner" style="width:18px;height:18px;margin:0;border-width:2px;"></div> Running...';

    showLoading(resultsPanel, getLoadingMessage());

    const attackType = $('#attack-type').value;
    const epsilon = parseFloat($('#epsilon-slider').value);

    const params = {
        attack_type: attackType,
        epsilon: epsilon,
        image_index: state.selectedImage.index,
        target_model: 'base',
    };

    // Add attack-specific params
    if (attackType === 'pgd') {
        params.steps = parseInt($('#pgd-steps')?.value || 40);
    } else if (attackType === 'genetic') {
        params.pop_size = parseInt($('#ga-pop-size')?.value || 30);
        params.generations = parseInt($('#ga-generations')?.value || 50);
    } else if (attackType === 'de') {
        params.maxiter = parseInt($('#de-maxiter')?.value || 50);
    }

    try {
        const res = await fetch('/api/attack', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params),
        });

        const data = await res.json();

        if (data.success) {
            displayAttackResults(data);
        } else {
            alert('Attack failed: ' + (data.error || 'Unknown error'));
        }
    } catch (err) {
        console.error('Attack error:', err);
        alert('Failed to run attack: ' + err.message);
    } finally {
        hideLoading(resultsPanel);
        btn.disabled = false;
        btn.innerHTML = '⚡ Run Attack';
        state.attackRunning = false;
    }
}

function getLoadingMessage() {
    const type = $('#attack-type')?.value;
    const messages = {
        fgsm: 'Computing gradient sign perturbation...',
        pgd: 'Running iterative projected gradient descent...',
        genetic: 'Evolving adversarial population (this may take a moment)...',
        de: 'Running differential evolution optimization...',
    };
    return messages[type] || 'Generating adversarial example...';
}

function displayAttackResults(data) {
    // ── Images ──
    $('#adv-image').innerHTML = `<img src="data:image/png;base64,${data.adversarial_image}" alt="Adversarial">`;
    $('#pert-image').innerHTML = `<img src="data:image/png;base64,${data.perturbation_image}" alt="Perturbation">`;

    // Labels
    const advLabel = $('#adv-label');
    if (advLabel) {
        const statusClass = data.attack_success ? 'danger' : 'success';
        const statusText = data.attack_success ? '✗ Misclassified' : '✓ Still Correct';
        advLabel.innerHTML = `Prediction: ${data.adv_pred} <span class="badge ${statusClass}">${statusText}</span>`;
    }

    // ── Confidence Bars ──
    updateConfidenceBars('orig', data.orig_probs, data.true_label, data.orig_pred);
    updateConfidenceBars('adv', data.adv_probs, data.true_label, data.adv_pred);

    // ── Metrics ──
    const metricsRow = $('#attack-metrics');
    if (metricsRow) {
        metricsRow.innerHTML = `
            <div class="metric-pill">
                <span class="metric-label">L∞</span>
                <span class="metric-value">${data.l_inf.toFixed(4)}</span>
            </div>
            <div class="metric-pill">
                <span class="metric-label">L2</span>
                <span class="metric-value">${data.l2.toFixed(4)}</span>
            </div>
            <div class="metric-pill">
                <span class="metric-label">Time</span>
                <span class="metric-value">${data.time.toFixed(3)}s</span>
            </div>
            ${data.queries ? `<div class="metric-pill"><span class="metric-label">Queries</span><span class="metric-value">${data.queries}</span></div>` : ''}
            ${data.generations ? `<div class="metric-pill"><span class="metric-label">Generations</span><span class="metric-value">${data.generations}</span></div>` : ''}
        `;
    }

    // ── Defense Results ──
    displayDefenseResults(data.defense_results, data.true_label);
}

function updateConfidenceBars(prefix, probs, trueLabel, predLabel) {
    for (let i = 0; i < 10; i++) {
        const fill = $(`#${prefix}-bar-${i}`);
        const val = $(`#${prefix}-val-${i}`);
        if (!fill || !val) continue;

        const pct = (probs[i] * 100).toFixed(1);
        fill.style.width = `${pct}%`;
        val.textContent = `${pct}%`;

        // Color: green if correct, red if wrong prediction, blue otherwise
        fill.className = 'bar-fill';
        if (i === trueLabel && i === predLabel) {
            fill.classList.add('predicted');
        } else if (i === predLabel && predLabel !== trueLabel) {
            fill.classList.add('wrong');
        } else {
            fill.classList.add('blue');
        }
    }
}

function displayDefenseResults(defenseResults, trueLabel) {
    const grid = $('#defense-results');
    if (!grid) return;

    const defenses = [
        { key: 'adv_training', name: 'Adversarial Training', icon: '🛡️' },
        { key: 'input_transform', name: 'Input Transform', icon: '🔄' },
        { key: 'detection', name: 'Detection Network', icon: '🔍' },
        { key: 'distillation', name: 'Distillation', icon: '🧪' },
    ];

    grid.innerHTML = defenses.map(def => {
        const res = defenseResults[def.key];
        if (!res) return '';

        let blocked, resultText;
        if (def.key === 'detection') {
            blocked = res.detected;
            resultText = res.detected
                ? `Detected (${(res.detection_confidence * 100).toFixed(0)}%)`
                : 'Not Detected';
        } else {
            blocked = res.correct;
            resultText = res.correct
                ? `Correct → ${res.prediction}`
                : `Fooled → ${res.prediction}`;
        }

        return `
            <div class="card defense-card">
                <div class="defense-status">${def.icon}</div>
                <div class="defense-name">${def.name}</div>
                <div class="defense-result ${blocked ? 'blocked' : 'bypassed'}">
                    ${blocked ? '✓ Blocked' : '✗ Bypassed'}
                </div>
                <div style="font-size:0.7rem;color:var(--text-tertiary);margin-top:6px">${resultText}</div>
            </div>
        `;
    }).join('');
}

// ─── Comparison Dashboard ───

let charts = {};

async function loadComparisonResults() {
    const container = $('#comparison-content');
    if (!container) return;

    showLoading(container, 'Loading evaluation results...');

    try {
        const res = await fetch('/api/results');

        if (!res.ok) {
            hideLoading(container);
            container.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">📊</div>
                    <p>No evaluation results found.</p>
                    <p style="margin-top:0.5rem;font-size:0.8rem">Run <code>python train_all.py</code> first to generate results.</p>
                </div>
            `;
            return;
        }

        const data = await res.json();
        hideLoading(container);
        renderComparison(data);
    } catch (err) {
        hideLoading(container);
        console.error('Failed to load results:', err);
    }
}

function renderComparison(data) {
    renderHeatmap(data);
    renderBarChart(data);
    renderRadarChart(data);
    renderMetricsTable(data);
}

function renderHeatmap(data) {
    const container = $('#heatmap-container');
    if (!container) return;

    const attacks = Object.keys(data.results);
    const defenses = Object.keys(data.defense_names);
    const attackNames = data.attack_names;
    const defenseNames = data.defense_names;

    let html = '<table class="heatmap-table"><thead><tr><th></th>';
    defenses.forEach(d => { html += `<th>${defenseNames[d]}</th>`; });
    html += '</tr></thead><tbody>';

    attacks.forEach(atk => {
        html += `<tr><td class="row-header">${attackNames[atk]}</td>`;
        defenses.forEach(def => {
            const val = data.results[atk]?.[def]?.attack_success_rate || 0;
            const heatClass = getHeatClass(val);
            html += `<td class="${heatClass}" data-tooltip="ASR: ${val}%">${val.toFixed(1)}%</td>`;
        });
        html += '</tr>';
    });

    html += '</tbody></table>';
    container.innerHTML = html;
}

function getHeatClass(val) {
    if (val < 20) return 'heat-0';
    if (val < 40) return 'heat-1';
    if (val < 60) return 'heat-2';
    if (val < 75) return 'heat-3';
    if (val < 90) return 'heat-4';
    return 'heat-5';
}

function renderBarChart(data) {
    const canvas = $('#bar-chart');
    if (!canvas) return;

    if (charts.bar) charts.bar.destroy();

    const attacks = Object.keys(data.results);
    const defenses = Object.keys(data.defense_names);
    const defenseNames = data.defense_names;
    const attackNames = data.attack_names;

    const colors = [
        'rgba(0, 180, 255, 0.8)',
        'rgba(139, 92, 246, 0.8)',
        'rgba(6, 214, 160, 0.8)',
        'rgba(244, 63, 94, 0.8)',
    ];

    const datasets = attacks.map((atk, i) => ({
        label: attackNames[atk],
        data: defenses.map(def => data.results[atk]?.[def]?.robust_accuracy || 0),
        backgroundColor: colors[i],
        borderColor: colors[i].replace('0.8', '1'),
        borderWidth: 1,
        borderRadius: 4,
    }));

    charts.bar = new Chart(canvas, {
        type: 'bar',
        data: {
            labels: defenses.map(d => defenseNames[d]),
            datasets: datasets,
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: { display: true, text: 'Robust Accuracy Under Attack (%)', color: '#f0f0f8', font: { size: 14, weight: 600 } },
                legend: { labels: { color: '#8892b0', font: { size: 11 } } },
            },
            scales: {
                x: { ticks: { color: '#8892b0', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.03)' } },
                y: { beginAtZero: true, max: 100, ticks: { color: '#8892b0' }, grid: { color: 'rgba(255,255,255,0.05)' } },
            },
        },
    });
}

function renderRadarChart(data) {
    const canvas = $('#radar-chart');
    if (!canvas) return;

    if (charts.radar) charts.radar.destroy();

    const defenses = Object.keys(data.defense_names).filter(d => d !== 'none');
    const defenseNames = data.defense_names;
    const attacks = Object.keys(data.results);

    // Calculate defense effectiveness: avg robust accuracy across all attacks
    const defenseData = defenses.map(def => {
        const avgRobust = attacks.reduce((sum, atk) =>
            sum + (data.results[atk]?.[def]?.robust_accuracy || 0), 0) / attacks.length;
        return avgRobust;
    });

    // Also get clean accuracy
    const cleanData = defenses.map(def => {
        const attack0 = attacks[0];
        return data.results[attack0]?.[def]?.clean_accuracy || 0;
    });

    charts.radar = new Chart(canvas, {
        type: 'radar',
        data: {
            labels: defenses.map(d => defenseNames[d]),
            datasets: [
                {
                    label: 'Clean Accuracy',
                    data: cleanData,
                    borderColor: 'rgba(6, 214, 160, 0.8)',
                    backgroundColor: 'rgba(6, 214, 160, 0.1)',
                    pointBackgroundColor: 'rgba(6, 214, 160, 1)',
                    borderWidth: 2,
                },
                {
                    label: 'Avg Robust Accuracy',
                    data: defenseData,
                    borderColor: 'rgba(0, 180, 255, 0.8)',
                    backgroundColor: 'rgba(0, 180, 255, 0.1)',
                    pointBackgroundColor: 'rgba(0, 180, 255, 1)',
                    borderWidth: 2,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: { display: true, text: 'Defense Performance Comparison', color: '#f0f0f8', font: { size: 14, weight: 600 } },
                legend: { labels: { color: '#8892b0' } },
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    ticks: { color: '#5a6380', backdropColor: 'transparent', font: { size: 10 } },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    pointLabels: { color: '#8892b0', font: { size: 11 } },
                    angleLines: { color: 'rgba(255,255,255,0.05)' },
                },
            },
        },
    });
}

function renderMetricsTable(data) {
    const container = $('#metrics-table-container');
    if (!container) return;

    const attacks = Object.keys(data.results);
    const defenses = Object.keys(data.defense_names);
    const attackNames = data.attack_names;
    const defenseNames = data.defense_names;

    let html = `
        <div style="overflow-x:auto">
        <table class="heatmap-table" style="border-spacing:2px">
            <thead>
                <tr>
                    <th>Attack</th>
                    <th>Defense</th>
                    <th>Clean Acc</th>
                    <th>Robust Acc</th>
                    <th>ASR</th>
                    <th>Avg L∞</th>
                    <th>Avg Time</th>
                </tr>
            </thead>
            <tbody>
    `;

    attacks.forEach(atk => {
        defenses.forEach(def => {
            const r = data.results[atk]?.[def] || {};
            html += `
                <tr>
                    <td class="row-header">${attackNames[atk]}</td>
                    <td class="row-header">${defenseNames[def]}</td>
                    <td>${(r.clean_accuracy || 0).toFixed(1)}%</td>
                    <td>${(r.robust_accuracy || 0).toFixed(1)}%</td>
                    <td class="${getHeatClass(r.attack_success_rate || 0)}">${(r.attack_success_rate || 0).toFixed(1)}%</td>
                    <td>${(r.avg_l_inf || 0).toFixed(4)}</td>
                    <td>${(r.avg_time || 0).toFixed(3)}s</td>
                </tr>
            `;
        });
    });

    html += '</tbody></table></div>';
    container.innerHTML = html;
}

// ─── Attack Type Change Handling ───
function onAttackTypeChange() {
    const type = $('#attack-type')?.value;
    // Show/hide param groups
    $$('.param-group').forEach(el => el.style.display = 'none');
    const group = $(`#params-${type}`);
    if (group) group.style.display = 'block';
}

// ─── Initialize ───
document.addEventListener('DOMContentLoaded', () => {
    // Attack Lab init
    if ($('#image-gallery')) {
        loadSampleImages();
    }

    // Epsilon slider
    const epsilonSlider = $('#epsilon-slider');
    if (epsilonSlider) {
        epsilonSlider.addEventListener('input', updateEpsilonDisplay);
    }

    // Attack type change
    const attackSelect = $('#attack-type');
    if (attackSelect) {
        attackSelect.addEventListener('change', onAttackTypeChange);
        onAttackTypeChange();
    }

    // Comparison page init
    if ($('#comparison-content')) {
        loadComparisonResults();
    }

    // Page animations
    $$('.fade-in-up').forEach((el, i) => {
        el.style.animationDelay = `${i * 0.1}s`;
    });
});
