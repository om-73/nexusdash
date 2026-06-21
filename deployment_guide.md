# Deployment Guide

This project is configured to be deployed as a **Single Web Service** on **Render** (Monorepo Style).

## 1. Push Code to GitHub
Ensure all recent changes are pushed to your `main` branch.

## 2. Deploy on Render (Recommended)

1.  Create a **New Web Service** connected to your repo.
2.  **Settings**:
    *   **Root Directory**: Leave empty (uses root `package.json`).
    *   **Runtime**: Node
    *   **Build Command**: `npm run build`
    *   **Start Command**: `npm start`
3.  **Environment Variables**:
    *   `PYTHON_ENGINE_URL`: URL of the Python service (if deployed separately) or `http://127.0.0.1:8000` if running locally in container (advanced).
    *   *Note: If you need the Python Engine, you should deploy the `data_engine` folder as a separate Python service and link it here.*

## 3. Alternative: Separate Services (Recommended by User)

### Frontend (Vercel)
1.  **Dashboard**: New Project > Import Git Repo.
2.  **Root Directory**: Click Edit > Select `client`.
3.  **Framework**: Vite (Automatic).
4.  **Environment Variables**:
    *   `VITE_API_URL`: `https://nexusdash-1.onrender.com/api`
    *   *(Note: I appended `/api` because your app routes are likely under /api)*
5.  **Deploy**: Click Deploy.
    *   *Note: `vercel.json` is already included to handle routing.*

### Backend (Render)
1.  **Dashboard**: New Web Service.
2.  **Root Directory**: `server`.
3.  **Build Command**: `npm install`.
4.  **Start Command**: `npm start`.


## Troubleshooting

### Authentication Resets
- **Issue**: "Authentication Failed" or users disappear after deployment.
- **Cause**: The app uses a local file (`models/users.json`) to store users. On Render, **files are wiped** on every deploy.
- **Solution**: You must **Register** a new account after every deployment.
- **Long-term**: Connect a database like Supabase or MongoDB.

### Vercel / Render Errors
- **"No Output Directory"**: Ensure `vercel.json` is present (it is in the latest code).
- **"Cannot find module"**: Ensure `package.json` has flattened dependencies (it is in the latest code).
- **CORS Errors**: If frontend fails to connect, check the Console. Ensure Backend URL is correct in Vercel Env Vars.
