import express from 'express';
import { createServer } from 'http';
import { Server } from 'socket.io';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const httpServer = createServer(app);
const io = new Server(httpServer, {
  cors: { origin: '*' }
});

// Serve the static files from the Vite build
app.use(express.static(path.join(__dirname, 'dist')));
app.use(express.json()); // Enable JSON parsing for incoming telemetry

// In-memory state for tasks
const taskState = {};

let lastTelemetryTime = 0;
const OFFLINE_THRESHOLD_MS = 3000;

// Watchdog timer to detect offline state
setInterval(() => {
    if (Date.now() - lastTelemetryTime > OFFLINE_THRESHOLD_MS) {
        io.emit('telemetry_offline');
    }
}, 1000);

io.on('connection', (socket) => {
    // Send initial state to the new client
    socket.emit('initialState', taskState);

    socket.on('taskUpdated', (data) => {
        taskState[data.taskId] = data.checked;
        // Broadcast the update to all OTHER connected clients
        socket.broadcast.emit('taskSync', data);
    });
});

// Endpoint for Python Telemetry Streamer
app.post('/api/telemetry', (req, res) => {
    lastTelemetryTime = Date.now();
    const telemetryData = req.body;
    io.emit('telemetry_update', telemetryData);
    res.status(200).json({ status: 'success', message: 'Telemetry broadcasted' });
});

const PORT = process.env.PORT || 3000;
httpServer.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
