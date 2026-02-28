# SmartExplain AI – Next-Level Frontend

## Stack

- **React 19** + **Vite 7** + **TypeScript**
- **Tailwind CSS** – styling
- **Recharts** – charts
- **Leaflet + React-Leaflet** – map
- **Lucide React** – icons
- **FastAPI** – backend API

## Run the Full Stack

### 1. Start the API backend

```bash
cd T:\ML Project\SmartExplain-AI
pip install fastapi uvicorn
python -m uvicorn api.main:app --reload --port 8000
```

### 2. Start the frontend

```bash
cd T:\ML Project\SmartExplain-AI\frontend
npm install
npm run dev
```

Then open **http://localhost:5173** in your browser.

## Features

1. **Prediction Playground** – Sliders for all House Prices features, live prediction, and contribution chart
2. **Explainability Hub** – Waterfall-style chart of feature contributions (green = increases price, red = decreases)
3. **What-If Lab** – Compare two scenarios (A vs B), see price difference
4. **Map Explorer** – California housing data on a map (when using `housing.csv`)
5. **Dark/Light Theme** – Toggle in the header

## Build for Production

```bash
cd frontend
npm run build
```

Output is in `frontend/dist/`.
