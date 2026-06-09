# Codementor AI Resource Retrieval Engine

This repository contains AI Intelligence and Resource Retrieval module of the **Codementor AI** website project. It functions as an autonomous **Resource Retrieval Engine** that converts abstract learning objectives into personalized, high-quality technical roadmaps.

---

## Website Architecture & The Roadmap Generator's Role

### How the Website Works:

1. **Code Analysis:** When a user submits code, an external analyzer scans the code for bugs, logic errors, or anti-patterns.
2. **Issue Mapping:** The diagnostic system maps each discovered issue to a specific, high-level learning topic (e.g., mapping a memory leak to `"Memory Management in C++"`).
3. **Resource Retrieval (This Code):** The website's frontend sends a POST request containing the programming language and the list of these mapped topics to the `/generate-roadmap` endpoint.

### What the Roadmap Generator Does:

For each topic received, this engine curates educational resources by orchestrating two parallel discovery pipelines:

* **Written Tutorials:** It queries a proprietary vector-encoded database (`links_with_embeddings.json`) containing self-scraped, pre-encoded technical documentation (e.g., GeeksforGeeks, W3Schools) to find contextually relevant text guides.
* **Video Resources:** It autonomously queries YouTube, applies strict linguistic and relevance filters, and checks community engagement to return authoritative video/playlist instruction.

---

## AI Backend (Hugging Face Space)

The resource retrieval engine is hosted and running autonomously on Hugging Face Spaces:
<img width="1264" height="482" alt="image" src="https://github.com/user-attachments/assets/d391a67e-b431-4ace-8faf-49d7475af8ea" />

---

## Interactive Roadmap (Web UI)

Here is an example of how the generated curriculum and resources are displayed on the frontend dashboard:
<img width="1525" height="666" alt="image" src="https://github.com/user-attachments/assets/6b9be269-0384-441e-88ea-2f6fa9a384f8" />

---

## Detailed Code Explanation

The FastAPI application processes requests linearly via modular functions:

### 1. Model & Tutorial Database Initialization

* **`SentenceTransformer("all-MiniLM-L6-v2")`**: Loads an NLP model used to generate vector embeddings for semantic searches.
* **`links_with_embeddings.json`**: A custom, self-scraped and embedded database containing tutorial titles, URLs, and pre-computed embeddings.

### 2. Written Tutorial Retrieval (`get_tutorials_links`)

* **`normalize_language`**: Maps non-standard or shorthand language names (e.g., `py`, `js`, `c plus plus`) to a unified format.
* **Layer 0 (Exact Match)**: Strips punctuation and language variants to check for direct matches between the requested topic and the tutorial database.
* **Layer 1 (Semantic Search)**: Computes the cosine similarity between the query embedding and the database embeddings using `SentenceTransformers`. If the `threshold` (0.55) is met, it returns the top matching tutorial URL.

### 3. Resilient YouTube Client Architecture

* **Secret Management & Key Rotation**: Fetches a comma-separated string of API keys from the environment variables (`YOUTUBE_API_KEYS`).
* **`get_youtube_client` & `switch_api_key**`: Instantiates a Google API client. If a 403 `HttpError` (Quota Exceeded) occurs during execution, the `safe_execute` wrapper catches the error, rotates to the next available API key in the list, and retries the request seamlessly.

### 4. YouTube Curation & Validation (`is_valid_result`)

* **`youtube_language_match`**: A hard keyword filter that confirms the target programming language string is present in the video title or description.
* **Language Detection**: Employs the `langdetect` library to analyze the combined title and description text. If the text is too short (< 10 chars) or not detected as English (`'en'`), the result is discarded.
* **Engagement Tiers**: Ensures scraped videos and playlists meet minimum view and like thresholds depending on the result tier to guarantee content authority.

### 5. Orchestration & API Endpoints

* **`get_youtube_links`**: Searches YouTube for both videos and playlists matching the query, pre-encodes the snippets, filters them through `is_valid_result`, validates them against engagement/similarity thresholds, and sorts by highest similarity.
* **`generate_roadmap`**: Loops through the incoming topics list, invokes `get_youtube_links` and `get_tutorials_links` for each, and structures the output into a nested JSON curriculum.
* **FastAPI endpoints (`/`, `/generate-roadmap`)**: Exposes the service via a POST endpoint that validates input payloads using the `GeminiInput` Pydantic model and runs a local Uvicorn server on port 7860.
