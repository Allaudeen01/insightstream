# InsightStream - Virtual Data Scientist

A SaaS application that transforms raw data into actionable business insights for non-technical users.

## 🚀 Quick Start

### Prerequisites
- Node.js 18+
- Python 3.10+

### Local Development

**1. Backend (FastAPI + Polars)**
```bash
cd engine
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
uvicorn main:app --reload
```

**2. Frontend (Next.js)**
```bash
cd web
npm install
npm run dev
```

Visit `http://localhost:3000`

---

## 🌐 Deployment

### Frontend → Vercel

1. Push code to GitHub
2. Go to [vercel.com](https://vercel.com) → New Project
3. Import your repository
4. Set root directory: `web`
5. Add environment variable:
   - `NEXT_PUBLIC_API_URL` = your backend URL
6. Deploy!

### Backend → Railway (Recommended)

1. Go to [railway.app](https://railway.app)
2. New Project → Deploy from GitHub
3. Select repository, set root: `engine`
4. Railway auto-detects Python
5. Add start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Copy the deployed URL → use in Vercel env vars

### Alternative: Backend → Render

1. Go to [render.com](https://render.com)
2. New → Web Service
3. Connect repo, root: `engine`
4. Build: `pip install -r requirements.txt`
5. Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`

---

## 📁 Project Structure

```
insightstream_-ai-data-analyst/
├── engine/                 # Python Backend
│   ├── main.py            # FastAPI application (9 endpoints)
│   ├── requirements.txt   # Python dependencies
│   └── venv/              # Virtual environment
├── web/                    # Next.js Frontend
│   ├── app/               # App router pages
│   │   ├── page.tsx       # Landing page
│   │   ├── upload/        # Screen 2: Upload
│   │   ├── health-check/  # Screen 3: Data Quality
│   │   ├── eda/           # Screen 4: Auto EDA
│   │   ├── insights/      # Screen 5: Insights
│   │   ├── chat/          # Screen 6: Chat
│   │   ├── modeling/      # Screen 7: AutoML
│   │   └── report/        # Screen 8: Reports
│   ├── vercel.json        # Vercel config
│   └── package.json
└── README.md
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload` | POST | Upload dataset |
| `/session/{id}` | GET | Get session info |
| `/health-check/{id}` | GET | Data quality analysis |
| `/clean/{id}` | POST | Auto-clean data |
| `/eda/{id}` | GET | Exploratory analysis |
| `/insights/{id}` | GET | Business insights |
| `/chat/{id}` | POST | NL queries |
| `/model/{id}` | POST | Train ML models |
| `/report/{id}` | GET | Generate report |

---

## 🛠 Tech Stack

- **Frontend:** Next.js 14, React 18, Tailwind CSS
- **Backend:** FastAPI, Polars, scikit-learn
- **Deployment:** Vercel (frontend), Railway/Render (backend)

---

## 📄 License

MIT
