import { io } from "socket.io-client";
import { marked } from "marked";

// Initialize socket connection
const socket = io();

document.addEventListener('DOMContentLoaded', () => {
    // --- TAB SWITCHING LOGIC ---
    const navItems = document.querySelectorAll('.nav-item');
    const tabPanes = document.querySelectorAll('.tab-pane');

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            // Remove active class from all nav items and tab panes
            navItems.forEach(nav => nav.classList.remove('active'));
            tabPanes.forEach(pane => pane.classList.remove('active'));

            // Add active class to clicked item and corresponding pane
            item.classList.add('active');
            const targetId = item.getAttribute('data-target');
            document.getElementById(targetId).classList.add('active');
        });
    });

    // --- REAL-TIME CHECKBOX STATE (SOCKET.IO) ---
    const checkboxes = document.querySelectorAll('.task input[type="checkbox"]');
    
    // Listen for initial state from server
    socket.on('initialState', (state) => {
        checkboxes.forEach((cb, index) => {
            if (state[index] !== undefined) {
                cb.checked = state[index];
            }
        });
    });

    // Listen for real-time updates from other clients
    socket.on('taskSync', (data) => {
        if (checkboxes[data.taskId]) {
            checkboxes[data.taskId].checked = data.checked;
        }
    });
    
    // Broadcast state on change
    checkboxes.forEach((cb, index) => {
        cb.addEventListener('change', () => {
            socket.emit('taskUpdated', {
                taskId: index,
                checked: cb.checked
            });
        });
    });

    // --- MARKDOWN VIEWER LOGIC ---
    const mdLinks = document.querySelectorAll('a[href$=".md"]');
    const mdModal = document.getElementById('md-modal');
    const mdCloseBtn = document.getElementById('md-close-btn');
    const mdContent = document.getElementById('md-content');
    const mdTitle = document.getElementById('md-modal-title');

    mdLinks.forEach(link => {
        link.addEventListener('click', async (e) => {
            e.preventDefault();
            const mdUrl = link.getAttribute('href');
            const fileName = mdUrl.split('/').pop();
            
            mdTitle.textContent = fileName;
            mdContent.innerHTML = '<p>Loading...</p>';
            mdModal.classList.remove('hidden');

            try {
                const response = await fetch(mdUrl);
                if (!response.ok) throw new Error('Network response was not ok');
                const mdText = await response.text();
                mdContent.innerHTML = marked.parse(mdText);
            } catch (error) {
                mdContent.innerHTML = `<p style="color: red;">Error loading ${fileName}: ${error.message}</p>`;
            }
        });
    });

    mdCloseBtn.addEventListener('click', () => {
        mdModal.classList.add('hidden');
    });

    // --- THEME TOGGLE LOGIC ---
    const themeToggleBtn = document.getElementById('theme-toggle');
    const currentTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', currentTheme);

    themeToggleBtn.addEventListener('click', () => {
        let theme = document.documentElement.getAttribute('data-theme');
        let newTheme = theme === 'light' ? 'dark' : 'light';
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
    });
});

// --- ARCHITECTURE ACCORDION LOGIC ---
window.toggleDetails = function(layerId) {
    const detailsDiv = document.getElementById(layerId);
    if (detailsDiv.classList.contains('hidden')) {
        detailsDiv.classList.remove('hidden');
    } else {
        detailsDiv.classList.add('hidden');
    }
}

// --- LIBRARY NAVIGATION LOGIC ---
window.openLibraryCategory = function(categoryId) {
    // Hide main grid
    document.getElementById('library-main-view').classList.add('hidden');
    
    // Show detail view container
    document.getElementById('library-detail-view').classList.remove('hidden');
    
    // Hide all individual categories
    const categories = document.querySelectorAll('.library-category');
    categories.forEach(cat => cat.classList.add('hidden'));
    
    // Show the target category
    document.getElementById(categoryId).classList.remove('hidden');
}

window.closeLibraryCategory = function() {
    // Hide detail view container
    document.getElementById('library-detail-view').classList.add('hidden');
    
    // Show main grid
    document.getElementById('library-main-view').classList.remove('hidden');
}

// --- MISSION CONTROL DASHBOARD LOGIC ---
document.addEventListener('DOMContentLoaded', () => {
    const ctx = document.getElementById('telemetryChart');
    if (!ctx) return; 
    
    // UI Elements
    const offlineOverlay = document.getElementById('offline-overlay');
    const liveIndicator = document.getElementById('live-indicator');
    const gaugeComp = document.getElementById('gauge-comp');
    const gaugeComb = document.getElementById('gauge-comb');
    const gaugeTurb = document.getElementById('gauge-turb');
    const hudThrust = document.getElementById('hud-thrust');
    const hudTsfc = document.getElementById('hud-tsfc');
    const hudPhysics = document.getElementById('hud-physics');
    
    const themeColor = '#0ea5e9'; // Sky Blue
    
    const telemetryChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Overall Engine Health (%)',
                data: [],
                borderColor: themeColor,
                backgroundColor: 'rgba(14, 165, 233, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            },
            {
                label: 'Upper Confidence Bound',
                data: [],
                borderColor: 'rgba(14, 165, 233, 0.3)',
                borderDash: [5, 5],
                fill: false,
                pointRadius: 0
            },
            {
                label: 'Lower Confidence Bound',
                data: [],
                borderColor: 'rgba(14, 165, 233, 0.3)',
                borderDash: [5, 5],
                fill: '-1', 
                backgroundColor: 'rgba(14, 165, 233, 0.05)',
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 },
            scales: {
                x: {
                    display: true,
                    title: { display: true, text: 'Time (Cycles)', color: '#64748b' },
                    grid: { color: 'rgba(14, 165, 233, 0.1)' },
                    ticks: { color: themeColor, font: { family: 'DotGothic16' } }
                },
                y: {
                    display: true,
                    title: { display: true, text: 'Health (%)', color: '#64748b' },
                    grid: { color: 'rgba(14, 165, 233, 0.1)' },
                    ticks: { color: themeColor, font: { family: 'DotGothic16' } },
                    min: 0,
                    max: 100
                }
            },
            plugins: {
                legend: { labels: { color: themeColor } }
            }
        }
    });

    const maxDataPoints = 50;

    socket.on('telemetry_update', (data) => {
        // Hide offline overlay
        if (offlineOverlay) offlineOverlay.classList.add('hidden');
        if (liveIndicator) {
            liveIndicator.textContent = "LIVE TELEMETRY FEED ACTIVE";
            liveIndicator.style.color = "#4ade80"; // Green
        }
        
        // Update Chart
        telemetryChart.data.labels.push(data.cycle);
        telemetryChart.data.datasets[0].data.push(data.overall_health);
        telemetryChart.data.datasets[1].data.push(data.overall_health + data.uncertainty_overall);
        telemetryChart.data.datasets[2].data.push(data.overall_health - data.uncertainty_overall);
        
        if (telemetryChart.data.labels.length > maxDataPoints) {
            telemetryChart.data.labels.shift();
            telemetryChart.data.datasets[0].data.shift();
            telemetryChart.data.datasets[1].data.shift();
            telemetryChart.data.datasets[2].data.shift();
        }
        telemetryChart.update();

        // Update Gauges
        const setGauge = (element, value) => {
            if (!element) return;
            const valNum = Math.round(value);
            element.innerText = valNum + '%';
            // Conic gradient for progress
            const color = valNum > 75 ? '#4ade80' : (valNum > 40 ? '#f59e0b' : '#ef4444');
            element.style.background = `conic-gradient(${color} ${valNum}%, var(--bg-main) 0%)`;
        };
        
        setGauge(gaugeComp, data.comp_health);
        setGauge(gaugeComb, data.comb_health);
        setGauge(gaugeTurb, data.turb_health);
        
        // Update Stats
        if (hudThrust) hudThrust.innerText = Math.round(data.thrust) + " N";
        if (hudTsfc) hudTsfc.innerText = data.tsfc.toFixed(4);
        if (hudPhysics) hudPhysics.innerText = data.physics_score;
    });
    
    socket.on('telemetry_offline', () => {
        if (offlineOverlay) offlineOverlay.classList.remove('hidden');
        if (liveIndicator) {
            liveIndicator.textContent = "CONNECTION LOST";
            liveIndicator.style.color = "#ef4444"; // Red
        }
    });

    // Initialize Violation Chart
    const vCtx = document.getElementById('violationChart');
    if (vCtx) {
        new Chart(vCtx, {
            type: 'bar',
            data: {
                labels: ['Icarus (GRU)', 'Titan (XGBoost)', 'Our PINN'],
                datasets: [{
                    label: 'Thermodynamic Violation Rate (%)',
                    data: [42.5, 38.2, 0.0],
                    backgroundColor: [
                        'rgba(239, 68, 68, 0.5)', // red
                        'rgba(245, 158, 11, 0.5)', // yellow
                        'rgba(74, 222, 128, 0.8)'  // green
                    ],
                    borderColor: [
                        '#ef4444',
                        '#f59e0b',
                        '#4ade80'
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
                        grid: { color: 'rgba(255, 124, 155, 0.1)' },
                        ticks: { color: '#666', font: { family: 'DotGothic16' } }
                    },
                    x: {
                        grid: { color: 'rgba(255, 124, 155, 0.1)' },
                        ticks: { color: '#666', font: { family: 'DotGothic16' } }
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }
});
