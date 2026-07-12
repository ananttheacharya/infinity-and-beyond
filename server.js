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

// In-memory state for tasks
const taskState = {};

io.on('connection', (socket) => {
    // Send initial state to the new client
    socket.emit('initialState', taskState);

    socket.on('taskUpdated', (data) => {
        taskState[data.taskId] = data.checked;
        // Broadcast the update to all OTHER connected clients
        socket.broadcast.emit('taskSync', data);
    });
});

const PORT = process.env.PORT || 3000;
httpServer.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
