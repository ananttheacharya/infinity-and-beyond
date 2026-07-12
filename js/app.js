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
