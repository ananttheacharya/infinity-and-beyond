const socket = io();

// UI Elements
const els = {
    connDot: document.getElementById('connection-dot'),
    connText: document.getElementById('connection-text'),
    cycle: document.getElementById('cycle-val'),
    overallHealth: document.getElementById('overall-health-val'),
    uncertainty: document.getElementById('uncertainty-val'),
    compHealth: document.getElementById('comp-health-val'),
    combHealth: document.getElementById('comb-health-val'),
    turbHealth: document.getElementById('turb-health-val'),
    physicsScore: document.getElementById('physics-score-val'),
    thrust: document.getElementById('thrust-val'),
    tsfc: document.getElementById('tsfc-val'),
    alt: document.getElementById('op-alt'),
    mach: document.getElementById('op-mach'),
    tamb: document.getElementById('op-tamb'),
    pamb: document.getElementById('op-pamb'),
    rpm: document.getElementById('op-rpm'),
    fuel: document.getElementById('op-fuel'),
    speedup: document.getElementById('speedup-val'),
    latency: document.getElementById('latency-val')
};

// Chart Setup
const ctx = document.getElementById('violationChart').getContext('2d');
const violationChart = new Chart(ctx, {
    type: 'bar',
    data: {
        labels: ['Baseline-PhysFeat', 'Baseline-Raw', 'Full Model (PCMN)'],
        datasets: [{
            label: 'Violation %',
            data: [0, 0, 0],
            backgroundColor: [
                'rgba(255, 99, 132, 0.6)',
                'rgba(255, 159, 64, 0.6)',
                'rgba(0, 210, 255, 0.6)'
            ],
            borderColor: [
                'rgba(255, 99, 132, 1)',
                'rgba(255, 159, 64, 1)',
                'rgba(0, 210, 255, 1)'
            ],
            borderWidth: 1
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            y: { 
                beginAtZero: true, 
                max: 100,
                grid: { color: 'rgba(255,255,255,0.1)' },
                ticks: { color: '#888' }
            },
            x: {
                grid: { display: false },
                ticks: { color: '#888' }
            }
        },
        plugins: {
            legend: { display: false }
        }
    }
});

// Load Benchmark JSON
fetch('/data/benchmark_results.json')
    .then(res => res.json())
    .then(data => {
        if(data && data.surrogate) {
            els.speedup.textContent = data.surrogate.speedup_x.toFixed(1) + 'x';
            els.latency.textContent = data.surrogate.surrogate_ms.toFixed(3) + ' ms/sample';
        }
    })
    .catch(err => console.log('No benchmark data yet'));

// Socket Events
socket.on('connect', () => {
    els.connDot.className = 'dot online';
    els.connText.textContent = 'LIVE STREAM';
});

socket.on('disconnect', () => {
    els.connDot.className = 'dot offline';
    els.connText.textContent = 'OFFLINE';
});

socket.on('telemetry_offline', () => {
    els.connDot.className = 'dot offline';
    els.connText.textContent = 'STREAM IDLE';
});

socket.on('telemetry_update', (data) => {
    els.connDot.className = 'dot online';
    els.connText.textContent = 'LIVE STREAM';
    
    // Update basic metrics
    els.cycle.textContent = data.cycle;
    els.overallHealth.textContent = data.overall_health.toFixed(1) + '%';
    els.uncertainty.textContent = data.uncertainty_overall.toFixed(2);
    els.physicsScore.textContent = data.physics_score;
    
    // Subsystems
    els.compHealth.textContent = data.comp_health.toFixed(1) + '%';
    els.combHealth.textContent = data.comb_health.toFixed(1) + '%';
    els.turbHealth.textContent = data.turb_health.toFixed(1) + '%';
    
    // Performance
    els.thrust.textContent = data.thrust.toFixed(0);
    els.tsfc.textContent = data.tsfc.toFixed(5);
    
    // Inputs
    els.alt.textContent = data.op_altitude.toFixed(0);
    els.mach.textContent = data.op_mach.toFixed(2);
    els.tamb.textContent = data.op_tamb.toFixed(1);
    els.pamb.textContent = data.op_pamb.toFixed(0);
    els.rpm.textContent = data.op_rpm.toFixed(0);
    els.fuel.textContent = data.op_fuel.toFixed(3);
    
    // Update Chart
    violationChart.data.datasets[0].data = [
        data.icarus_violation,
        data.titan_violation,
        data.pinn_violation
    ];
    violationChart.update();
    
    // Update visual engine colors based on health
    updateEngineStage('.compressor-stage', data.comp_health);
    updateEngineStage('.combustor-stage', data.comb_health);
    updateEngineStage('.turbine-stage', data.turb_health);
});

function updateEngineStage(selector, health) {
    const el = document.querySelector(selector);
    if (!el) return;
    
    let color = 'rgba(0, 255, 136, 0.3)'; // Green
    if (health < 70) color = 'rgba(255, 159, 64, 0.4)'; // Orange
    if (health < 40) color = 'rgba(255, 51, 102, 0.5)'; // Red
    
    el.style.backgroundColor = color;
}
