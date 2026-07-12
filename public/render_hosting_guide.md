# Deployment Guide: Hosting the AEROTHON Digital Twin Dashboard on Render

This guide outlines the steps to host this exact dashboard (Vite + Node.js/Socket.io backend) on **Render.com**, a popular and free-tier-friendly cloud platform.

## Why Render?
- Render natively supports Node.js applications.
- It provides free HTTPS, automatic GitHub deployments, and WebSocket support out-of-the-box, which is critical for our `socket.io` real-time task syncing feature.

---

## Prerequisites

Before starting, ensure you have:
1. Pushed the entire dashboard project (including `package.json`, `server.js`, `vite.config.js` if any, `index.html`, etc.) to a **GitHub Repository**.
2. Created a free account on [Render.com](https://render.com/).

---

## Step-by-Step Deployment

### 1. Create a New Web Service
1. Log in to your Render dashboard.
2. Click the **"New"** button in the top right corner.
3. Select **"Web Service"**.

### 2. Connect Your GitHub Repository
1. On the "Create Web Service" page, choose **"Build and deploy from a Git repository"**.
2. Connect your GitHub account if you haven't already.
3. Search for the repository containing this dashboard project and click **"Connect"**.

### 3. Configure the Build Settings
Render needs to know how to build the Vite frontend and run the Node.js backend. Fill in the following details:

- **Name:** Choose a name (e.g., `aerothon-dashboard`).
- **Region:** Choose the region closest to your location or the judges' location (e.g., `Singapore` or `Frankfurt`).
- **Branch:** `main` (or whichever branch you are using).
- **Runtime:** `Node`

**Critical Build Commands:**
- **Build Command:** `npm install && npm run build`
  *(This installs both backend and frontend dependencies, then runs Vite to generate the static `dist/` folder for the frontend).*
- **Start Command:** `npm start`
  *(This must map to `node server.js` in your `package.json`, which serves the `dist/` folder and starts the Socket.io server).*

### 4. Configure Environment Variables (If needed)
If you add API keys later (e.g., for external ML inferences), you can add them by clicking **"Advanced" -> "Add Environment Variable"**. Currently, this static-and-socket dashboard doesn't require extra keys.

### 5. Deploy
1. Scroll down and select the **"Free"** instance type.
2. Click **"Create Web Service"**.

Render will now clone your repo, run the build command, and start the Node server. You will see a terminal output displaying the progress.

---

## 6. Verification & Usage
Once the deployment finishes (usually 2-3 minutes), Render will provide a URL at the top left of the screen (e.g., `https://aerothon-dashboard.onrender.com`).

1. Open the URL.
2. Verify the Dark Mode toggle and Markdown Viewer work correctly.
3. **Test WebSockets:** Open the URL on your phone and your laptop simultaneously. Check a box in the "Workstreams" tab on your laptop, and verify it instantly checks off on your phone.

You are now live! Share this URL with your team and the AEROTHON judges.
