# V0 Frontend: Wire Up "Generate Legal Fiction"

If your V0-built "Generate Legal Fiction" button does nothing when clicked, it isn't connected to the backend API yet. Add the following to your V0 component.

## Backend URL

Replace `YOUR_BACKEND_URL` with your actual backend URL, e.g.:

- Local: `http://localhost:8000`
- Ngrok: `https://abc123.ngrok-free.app`

## React / Next.js Example

Add state and a click handler to your case detail component:

```tsx
const [isGenerating, setIsGenerating] = useState(false);
const [error, setError] = useState<string | null>(null);

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

async function handleGenerate(caseId: string) {
  setIsGenerating(true);
  setError(null);
  try {
    const res = await fetch(`${BACKEND_URL}/api/case/${encodeURIComponent(caseId)}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        mode: "courtroom_focused",
        target_length: "800-1200",
      }),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || "Generation failed");
    }
    // Success: refresh your stories list or navigate
    window.location.reload(); // or refetch stories, show toast, etc.
  } catch (err) {
    setError(err instanceof Error ? err.message : "Something went wrong");
  } finally {
    setIsGenerating(false);
  }
}
```

## Button with Progress

```tsx
<button
  onClick={() => handleGenerate(caseId)}
  disabled={isGenerating}
>
  {isGenerating ? "Generating… (2–5 min)" : "Generate Legal Fiction"}
</button>
{error && <p className="text-red-500">{error}</p>}
```

## API Contract

- **Endpoint:** `POST /api/case/{case_id}/generate`
- **Body (JSON):**
  ```json
  {
    "mode": "courtroom_focused",
    "target_length": "800-1200",
    "model_override": null
  }
  ```
- **Success (200):**
  ```json
  {
    "success": true,
    "story_id": 1,
    "case_id": "t17800628-12",
    "created_at": "2024-01-15T12:00:00",
    "compliant": true,
    "message": "Story generated."
  }
  ```
- **Error (404/502):**
  ```json
  {
    "success": false,
    "error": "Case not found"
  }
  ```

**Note:** Generation takes 2–5 minutes. Keep the loading state visible until the request completes.
