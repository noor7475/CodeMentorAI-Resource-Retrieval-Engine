import os
import re
import torch
import json
import uvicorn
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from sentence_transformers import SentenceTransformer, util
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Initialize Model
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

# Load json
with open("links_with_embeddings.json", "r") as f:
    db = json.load(f)

def normalize_language(language):
    mapping = {
        "c++": "cpp",
        "c plus plus": "cpp",
        "c#": "c-sharp",
        "c sharp": "c-sharp",
        "c_sharp": "c-sharp",
        "js": "javascript",
        "py": "python"
    }

    lang = language.lower().strip()
    return mapping.get(lang, lang)

# Tutorials function
def get_tutorials_links(language, query, threshold=0.55, top_k=1):
    lang = normalize_language(language)
    escaped_language = re.escape(lang)

    if lang not in db:
        return []

    lang_data = db[lang]
    query_clean = query.lower().strip()

    # --- LAYER 0: EXACT MATCH ---
    exact_matches = {}
    for full_title, data in lang_data.items():
        url = data['url']
        base_title = full_title.split('(')[0].strip()

        clean_title = re.sub(rf'\b{escaped_language}\b', '', base_title, flags=re.IGNORECASE).strip().lower()

        if query_clean == clean_title:
            source = "GFG" if "GFG" in full_title else "W3S" if "W3S" in full_title else "Video"
            exact_matches[source] = url

    if exact_matches:
      return list(exact_matches.values())

    # --- LAYER 1: SEMANTIC SEARCH ---
    query_emb = model.encode(query_clean, convert_to_tensor=True, normalize_embeddings=True)
    candidates = []

    for full_title, data in lang_data.items():
        clean_title = re.sub(rf'\b{escaped_language}\b', '', full_title.split('(')[0], flags=re.IGNORECASE).strip()
        title_emb = torch.tensor(data['embedding'], dtype=torch.float32)
        score = util.cos_sim(query_emb, title_emb).item()

        if score >= threshold:
            source = "GFG" if "GFG" in full_title else "W3S" if "W3S" in full_title else "Video"
            candidates.append({
                "topic": clean_title,
                "url": data['url'],
                "source": source,
                "score": score
            })

    if candidates:
        # Sort by similarity and keep top_k
        candidates.sort(key=lambda x: x['score'], reverse=True)
        top_candidates = candidates[:top_k]

        return [c['url'] for c in top_candidates]

    return []

# Initialize YouTube Client
keys_str = os.environ.get("YOUTUBE_API_KEYS", "")

YOUTUBE_API_KEYS = [k.strip() for k in keys_str.split(",") if k.strip()]

current_key_index = 0

def get_youtube_client():
    global current_key_index
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEYS[current_key_index])

youtube = get_youtube_client()

def switch_api_key():
    global current_key_index, youtube

    if current_key_index + 1 < len(YOUTUBE_API_KEYS):
        current_key_index += 1
        #print(f"[SWITCHING API KEY] Using key index {current_key_index}")
        youtube = get_youtube_client()
        return True
    else:
        print("[ALL API KEYS EXHAUSTED]")
        return False

def safe_execute(request):
    global youtube

    try:
        return request.execute()
    except HttpError as e:
        if e.resp.status == 403:  # Quota exceeded
            print("[QUOTA EXCEEDED] Trying next API key...")
            if switch_api_key():
                return request.execute()
        raise e

# Helper Functions
def get_video_stats(video_ids):
    """Fetches statistics for a batch of video IDs (up to 50)."""
    if not video_ids: return {}
    try:
        stats_res = youtube.videos().list(
            part="statistics",
            id=",".join(video_ids)
        ).execute()
        stats_map = {}
        for item in stats_res.get("items", []):
            s = item["statistics"]
            stats_map[item["id"]] = {
                "views": int(s.get("viewCount", 0)),
                "likes": int(s.get("likeCount", 0))
            }
        return stats_map
    except Exception as e:
        print(f"Error fetching video stats: {e}")
        return {}

def youtube_language_match(language, title):
    lang = language.lower().strip()
    title = title.lower()

    language_variants = {
        "python": ["python"],
        "javascript": ["javascript", "js"],
        "c": [" c "],
        "c++": ["c++", "cpp", "c plus plus"],
        "c#": ["c#", "csharp", "c sharp", "c-sharp", "c_sharp"],
        "java": ["java"]
    }

    variants = language_variants.get(lang, [lang])

    return any(v in title for v in variants)

# Seed to make language detection consistent
DetectorFactory.seed = 0

def is_valid_result(prog_lang, title, description):
    """
    Checks for programming language keyword AND ensures 
    the content is in English.
    """
    # 1. Check for the programming language
    has_prog_lang = youtube_language_match(prog_lang, title) or youtube_language_match(prog_lang, description)
    
    # 2. Check if it is actually in English
    full_text = f"{title} {description}"

    if len(full_text.strip()) < 10:
        return False 
    
    try:
        is_eng = detect(full_text) == 'en'
    except LangDetectException:
        # If detection fails, we cannot confirm it's English
        return False
    
    return has_prog_lang and is_eng
    
# Youtube Similarity Threshold
MIN_SIMILARITY = 0.5

# YouTube Scraper (Playlists & Videos)
def get_youtube_links(skill_area, language):
    query = f'{language} {skill_area}' 
    target_emb = model.encode(query)

    # Thresholds for Youtube Quality Control
    video_tiers = [(15000, 750), (5000, 250), (1000, 50)]
    playlist_tiers = [(2000, 100), (800, 40), (300, 15)]

    try:
        search_pl = safe_execute(youtube.search().list(part="id,snippet", q=query, type="playlist", maxResults=25))
        raw_playlists = search_pl.get("items", [])

        search_vid = safe_execute(youtube.search().list(part="id,snippet", q=query, type="video", maxResults=25, videoDuration="medium"))
        raw_videos = search_vid.get("items", [])
    except Exception as e:
        return {"playlists": [], "videos": []}

    # --- Pre-encode Metadata ---
    pl_meta = []
    for item in raw_playlists:
        t, d = item['snippet']['title'], item['snippet'].get("description", "")
        pl_meta.append({
            "id": item['id']['playlistId'],
            "title": t,
            "desc": d,
            "emb": model.encode(t + " " + d) # Encoded once!
        })

    vid_meta_list = []
    for item in raw_videos:
        t, d = item['snippet']['title'], item['snippet'].get("description", "")
        vid_meta_list.append({
            "id": item['id']['videoId'],
            "title": t,
            "desc": d,
            "emb": model.encode(t + " " + d) # Encoded once!
        })

    final_results = {"playlists": [], "videos": []}

    # --- INDEPENDENT PLAYLIST SEARCH ---
    for v_min, l_min in playlist_tiers:
        found_pl = []
        for item in pl_meta: # Using our pre-encoded list
            if not is_valid_result(language, item["title"], item["desc"]):
                continue
            
            sim = util.cos_sim(target_emb, item["emb"]).item()
            current_sim_floor = 0.5 if v_min > 1000 else 0.4
            
            if sim < current_sim_floor: continue

            # Stats check logic...
            pl_items = youtube.playlistItems().list(part="contentDetails", playlistId=item["id"], maxResults=8).execute()
            v_ids = [v['contentDetails']['videoId'] for v in pl_items.get("items", [])]
            if v_ids:
                stats = get_video_stats(v_ids)
                if not stats: continue
                avg_v = sum(s['views'] for s in stats.values()) / len(stats)
                avg_l = sum(s['likes'] for s in stats.values()) / len(stats)

                if avg_v >= v_min and avg_l >= l_min:
                    found_pl.append({
                        "title": item["title"], "url": f"https://www.youtube.com/playlist?list={item['id']}",
                        "avg_views": int(avg_v), "avg_likes": int(avg_l), "similarity": round(sim, 3)
                    })
        
        if found_pl:
            found_pl.sort(key=lambda x: x["similarity"], reverse=True)
            final_results["playlists"] = found_pl[:1]
            break 

    # --- INDEPENDENT VIDEO SEARCH ---
    v_ids_all = [v["id"] for v in vid_meta_list]
    all_stats = get_video_stats(v_ids_all)

    for v_min, l_min in video_tiers:
        found_vid = []
        for item in vid_meta_list:
            s = all_stats.get(item["id"])
            if not s: continue
            
            if not is_valid_result(language, item["title"], item["desc"]):
                continue
              
            sim = util.cos_sim(target_emb, item["emb"]).item()
            if sim >= MIN_SIMILARITY and s['views'] >= v_min and s['likes'] >= l_min:
                found_vid.append({
                    "title": item["title"], "url": f"https://www.youtube.com/watch?v={item['id']}",
                    "views": s['views'], "likes": s['likes'], "similarity": sim
                })

        if found_vid:
            found_vid.sort(key=lambda x: x["similarity"], reverse=True)
            final_results["videos"] = found_vid[:1]
            break 

    return final_results
  
# Roadmap generation
def generate_roadmap(gemini_output):
    language = gemini_output["language"]
    topics = gemini_output["topics"]

    roadmap = []

    for topic in topics:
        yt_data = get_youtube_links(topic, language)
        tutorials = get_tutorials_links(language, topic)

        roadmap.append({
            "topic": topic.title(),
            "resources": {
                "youtube": {
                    "video": [
                        {"title": v["title"], "url": v["url"]}
                        for v in yt_data["videos"][:1]
                    ] if yt_data["videos"] else None,
                    "playlist": [
                        {"title": p["title"], "url": p["url"]}
                        for p in yt_data["playlists"][:1]
                    ] if yt_data["playlists"] else None
                },
                "tutorials": tutorials if tutorials else []
            }
        })

    return {
        "language": language,
        "roadmap": roadmap
    }

# Create server
app = FastAPI()

# Optional: allow requests from anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input model for json
class GeminiInput(BaseModel):
    language: str
    topics: list[str]

@app.get("/")
def home():
    return {"message": "Roadmap API is running"}
    
@app.post("/generate-roadmap")
def generate_roadmap_api(data: GeminiInput):
    print(f"--- Incoming Request ---")
    print(f"Language: {data.language}")
    print(f"Topics: {data.topics}")
    print(f"------------------------")
    
    return generate_roadmap(data.dict())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)