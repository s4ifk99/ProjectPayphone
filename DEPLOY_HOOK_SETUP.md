# Vercel Deploy Hook Setup

This guide walks you through creating a deploy hook in Vercel and wiring it to your FastAPI backend so each generated legal fiction story triggers a fresh Vercel build.

---

## Step 1: Create the deploy hook in Vercel

1. Go to [vercel.com](https://vercel.com) and sign in.
2. Open the project that hosts your **V0 frontend** (the site that displays stories from `GET /api/stories`).
3. Go to **Settings** → **Git** → **Deploy Hooks**.
4. Under "Create Hook":
   - **Name**: e.g. `Legal fiction generated`
   - **Branch**: usually `main` (or whichever branch you deploy from)
5. Click **Create Hook**.
6. Copy the generated URL. It looks like:
   ```
   https://api.vercel.com/v1/integrations/deploy/XXXXXXXXXXXXXXXXXX/YYYYYYYYYYYYYYYY
   ```

---

## Step 2: Configure your FastAPI backend

Set the hook URL so the backend can call it after each story generation.

### Option A: Using a `.env` file (recommended)

1. Copy the example env file:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env` and add the deploy hook line with your real URL from Step 1:
   ```
   VERCEL_DEPLOY_HOOK_URL=https://api.vercel.com/v1/integrations/deploy/YOUR_HOOK_ID
   ```
3. The app loads `.env` automatically when it starts — just restart `uvicorn`.

### Option B: Inline when starting the app

```bash
VERCEL_DEPLOY_HOOK_URL="https://api.vercel.com/v1/integrations/deploy/YOUR_HOOK_ID" uvicorn app.main:app --reload
```

---

## Step 3: Verify

1. Install deps if needed: `pip install -r requirements.txt`
2. Start your FastAPI backend (it will read `.env` if present).
3. Generate a legal fiction story (via the UI or API).
4. After generation completes, check:
   - Your Vercel project **Deployments**: a new deployment should appear.
   - FastAPI logs: look for `Deploy hook triggered successfully.`

---

## Troubleshooting

| Issue | What to try |
|-------|-------------|
| No new deployment in Vercel | Confirm `VERCEL_DEPLOY_HOOK_URL` is set and the app was restarted after editing `.env`. |
| `Deploy hook failed` in logs | The hook URL may be wrong, or the network may block outbound requests. Test with `curl -X POST "YOUR_HOOK_URL"`. |
| V0 site not showing new story | Ensure the frontend fetches from your backend’s `/api/stories` (or equivalent) and that CORS is configured correctly. |
