import os
import requests
import sqlite3
import numpy as np
import json
import random
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Load Environment Variables ---
load_dotenv()
SERVER_URL = os.getenv("JELLYFIN_URL")
API_KEY = os.getenv("JELLYFIN_API_KEY")
DB_PATH = os.getenv("DB_PATH")

def create_item_fingerprint(item):
    """
    Converts a Jellyfin item into a weighted dictionary (Super-Fingerprint).
    Deduplicates tags that mirror genres or decades.
    """
    fingerprint = {}
    
    # Pre-process genres and decades for deduplication
    genres_lower = [g.lower() for g in item.get("Genres", [])]
    year = item.get("ProductionYear")
    decade_string = f"{(year // 10) * 10}s" if year else None
    
    # 1. Decade (Base Weight: 1.2)
    if decade_string:
        fingerprint[f"Decade:{decade_string}"] = 1.2
        
    # 2. Genres (Base Weight: 1.0)
    for genre in item.get("Genres", []):
        fingerprint[f"Genre:{genre}"] = 1.0
        
    # 3. Theme/Trope Tags (Base Weight: 1.5)
    for tag in item.get("Tags", []):
        tag_lower = tag.lower()
        # Skip this tag if it's already accounted for by a Genre or Decade
        if tag_lower in genres_lower or tag_lower == decade_string:
            continue
        fingerprint[f"Tag:{tag}"] = 1.5
        
    # 4. People (Base Weight: 0.8) - Expanded to 15 to capture ensemble casts
    people = item.get("People", [])[:15] 
    for person in people:
        fingerprint[f"Person:{person.get('Name')}"] = 0.8
        
    return fingerprint

def build_feature_matrix(items):
    """
    Builds a unified vector matrix for all media items.
    Returns the vectorizer, the matrix, and a dictionary mapping Item IDs to their matrix index.
    """
    fingerprints = []
    item_id_to_index = {}
    
    for idx, item in enumerate(items):
        fingerprints.append(create_item_fingerprint(item))
        item_id_to_index[item.get("Id")] = idx
        
    vectorizer = DictVectorizer(sparse=False)
    feature_matrix = vectorizer.fit_transform(fingerprints)
    
    return vectorizer, feature_matrix, item_id_to_index

def get_user_preference_vector(user_id, feature_matrix, item_id_to_index):
    """
    Calculates the rolling average vector of all items a user has fully watched.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Fetch IDs of items the user has completed or played
    cursor.execute('''
        SELECT item_id FROM watch_history 
        WHERE user_id = ? AND (completion_percentage = 100.0 OR play_count > 0)
    ''', (user_id,))
    
    watched_ids = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    # Find the corresponding rows in the feature matrix
    watched_indices = [item_id_to_index[i_id] for i_id in watched_ids if i_id in item_id_to_index]
    
    if not watched_indices:
        return None # User has no usable watch history
        
    # Isolate the user's watched media vectors
    watched_vectors = feature_matrix[watched_indices]
    
    # Calculate the rolling average (mean across columns)
    user_vector = np.mean(watched_vectors, axis=0)
    
    # Reshape to a 2D array (1 row, N columns) for scikit-learn compatibility later
    return user_vector.reshape(1, -1)

def get_gradient_recommendations(user_vector, unwatched_items, feature_matrix, item_id_to_index):
    """
    Calculates cosine similarity and selects 10 items using the 4-3-3 Exploit/Moderate/Explore gradient.
    """
    if not unwatched_items:
        return []

    # Filter out any items that didn't make it into the matrix
    valid_unwatched_items = [item for item in unwatched_items if item["Id"] in item_id_to_index]
    unwatched_indices = [item_id_to_index[item["Id"]] for item in valid_unwatched_items]
    
    if not unwatched_indices:
        return []

    # Extract the vectors for only the unwatched items
    unwatched_vectors = feature_matrix[unwatched_indices]
    
    # Calculate Cosine Similarity (Returns a 2D array, we grab the first row)
    similarities = cosine_similarity(user_vector, unwatched_vectors)[0] 
    
    # Pair items with their similarity score and sort descending
    scored_items = []
    for i, item in enumerate(valid_unwatched_items):
        scored_items.append({
            "Item": item,
            "Score": float(similarities[i])
        })
        
    scored_items.sort(key=lambda x: x["Score"], reverse=True)
    
    total_items = len(scored_items)
    
    # If the user has fewer than 10 unwatched items total, just return them all
    if total_items < 10:
        return scored_items 
        
    # Define our gradient pools based on library percentiles
    # Safe: Top 10% | Moderate: 40%-60% | Reach: 70%-90%
    safe_pool = scored_items[:max(4, int(total_items * 0.1))]
    moderate_pool = scored_items[int(total_items * 0.4):int(total_items * 0.6)]
    reach_pool = scored_items[int(total_items * 0.7):int(total_items * 0.9)]
    
    # Fallback safeties in case of extremely small unwatched libraries
    if not moderate_pool: moderate_pool = scored_items
    if not reach_pool: reach_pool = scored_items
    
    # Assemble the 4-3-3 Playlist
    gradient_playlist = []
    gradient_playlist.extend(safe_pool[:4]) # Strictly the top 4 highest matches
    gradient_playlist.extend(random.sample(moderate_pool, 3)) # 3 random mid-tier matches
    gradient_playlist.extend(random.sample(reach_pool, 3)) # 3 random low-tier exploratory matches
    
    return gradient_playlist

def apply_reversion_logic(gradient_playlist, unwatched_items):
    """
    Checks the 10 selected recommendations against 4 tiers of prerequisites 
    to ensure sequels are not recommended before unwatched prequels.
    """
    validated_top_10 = []
    seen_ids = set()
    
    for rec in gradient_playlist:
        item = rec["Item"]
        item_type = item.get("Type")
        
        if item_type == "Movie":
            rec_tags = item.get("Tags", [])
            rec_sort = item.get("SortName", item.get("Name", "")).lower()
            rec_year = item.get("ProductionYear", 9999)
            
            franchise_tags = [t for t in rec_tags if t.lower().startswith("franchise:")]
            universe_tags = [t for t in rec_tags if t.lower().startswith("universe:")]
            depends_tags = [t for t in rec_tags if t.lower().startswith("dependson:")]
            
            older_relatives = []
            
            # Tier 1: Franchise Match
            if franchise_tags:
                for target_tag in franchise_tags:
                    for m in unwatched_items:
                        if m.get("Type") == "Movie" and m.get("ProductionYear", 9999) < rec_year:
                            if target_tag in m.get("Tags", []):
                                older_relatives.append(m)
                                
            # Tier 2: Universe Match
            elif universe_tags:
                for target_tag in universe_tags:
                    for m in unwatched_items:
                        if m.get("Type") == "Movie" and m.get("ProductionYear", 9999) < rec_year:
                            if target_tag in m.get("Tags", []):
                                older_relatives.append(m)
                                
            # Tier 3: DependsOn Cross-Reference
            elif depends_tags:
                for target_tag in depends_tags:
                    base_name = target_tag.split(":", 1)[1].strip().lower()
                    for m in unwatched_items:
                        if m.get("Type") == "Movie" and m.get("ProductionYear", 9999) <= rec_year:
                            m_tags_lower = [t.lower() for t in m.get("Tags", [])]
                            if f"franchise: {base_name}" in m_tags_lower or f"universe: {base_name}" in m_tags_lower:
                                if m.get("Id") != item.get("Id"):
                                    older_relatives.append(m)

            # Tier 4: Sort Name Match
            else:
                for m in unwatched_items:
                    if m.get("Type") == "Movie" and m.get("ProductionYear", 9999) < rec_year:
                        m_sort = m.get("SortName", m.get("Name", "")).lower().strip()
                        rec_sort_clean = rec_sort.strip()
                        
                        if len(m_sort) > 3:
                            is_match = False
                            
                            # 1. Exact match before a colon
                            if ":" in rec_sort_clean and ":" in m_sort and rec_sort_clean.split(":")[0].strip() == m_sort.split(":")[0].strip():
                                is_match = True
                                
                            # 2. Strict Prefix Match
                            elif rec_sort_clean.startswith(m_sort):
                                remainder = rec_sort_clean[len(m_sort):].strip()
                                valid_roman_numerals = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]
                                
                                # Only allow specific sequel indicators
                                if not remainder:
                                    is_match = True
                                elif remainder[0] in [":", "-"]:
                                    is_match = True
                                elif remainder.isdigit():
                                    is_match = True
                                elif remainder in valid_roman_numerals:
                                    is_match = True
                                elif remainder in ["part 2", "part 3", "vol 2", "vol 3"]:
                                    is_match = True
                                    
                            if is_match:
                                older_relatives.append(m)
           
            # Apply Swap
            if older_relatives:
                older_relatives.sort(key=lambda x: x.get("ProductionYear", 9999))
                oldest_prequel = older_relatives[0]
                
                if oldest_prequel.get("Id") not in seen_ids:
                    print(f"Swapped sequel '{item.get('Name')}' for older entry '{oldest_prequel.get('Name')}'")
                    validated_top_10.append({
                        "Item": oldest_prequel,
                        "Score": rec["Score"],
                        "SwappedFrom": item.get("Name")
                    })
                    seen_ids.add(oldest_prequel.get("Id"))
                continue 
                
        # If it's a Series, or a Movie with no older relatives
        if item.get("Id") not in seen_ids:
            validated_top_10.append(rec)
            seen_ids.add(item.get("Id"))

    return validated_top_10

def log_to_database(user_id, username, interacted_items):
    """Logs watched items to preserve your existing watch history."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    current_watch_ticks = 0
    
    for item in interacted_items:
        item_id = item.get("Id")
        user_data = item.get("UserData", {})
        played = user_data.get("Played", False)
        play_count = user_data.get("PlayCount", 0)
        playback_ticks = user_data.get("PlaybackPositionTicks", 0)
        runtime_ticks = item.get("RunTimeTicks", 0)
        
        current_watch_ticks += playback_ticks if not played else runtime_ticks
        
        completion_percentage = 0.0
        if played or play_count > 0:
            completion_percentage = 100.0
        elif runtime_ticks and runtime_ticks > 0:
            completion_percentage = round((playback_ticks / runtime_ticks) * 100.0, 2)
        
        cursor.execute('''
            INSERT OR REPLACE INTO watch_history 
            (user_id, username, item_id, item_name, item_type, production_year, tags, community_rating, runtime_ticks, play_count, completion_percentage)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, username, item_id, item.get("Name"), item.get("Type"), item.get("ProductionYear", 0), 
              json.dumps(item.get("Tags", [])), item.get("CommunityRating", 0.0), runtime_ticks, play_count, completion_percentage))
        
    conn.commit()
    conn.close()
    return current_watch_ticks

def log_active_recommendations(user_id, validated_top_10, current_ticks):
    """Logs recommendations to track the 30-hour negative feedback loop."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for rec in validated_top_10:
        cursor.execute('''
            INSERT OR IGNORE INTO active_recommendations (user_id, item_id, user_watch_ticks_at_rec)
            VALUES (?, ?, ?)
        ''', (user_id, rec["Item"]["Id"], current_ticks))
    conn.commit()
    conn.close()

def initialize_markdown_log():
    """Appends a new run header to the markdown log."""
    log_path = os.getenv("LOG_PATH", "recommender_log.md")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n\n# =========================================\n")
        f.write(f"# Recommendation Engine Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("# =========================================\n\n")

def append_user_audit_log(username, user_vector, vectorizer, validated_top_10, newly_rejected_names):
    """Appends user specific weights, rejections, and reversion triggers to the markdown log."""
    log_path = os.getenv("LOG_PATH", "recommender_log.md")
    
    feature_names = vectorizer.get_feature_names_out()
    weights = user_vector[0]
    top_indices = np.argsort(weights)[::-1][:10]
    top_tags = [(feature_names[i], weights[i]) for i in top_indices if weights[i] > 0]
    
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"## Audit Log for {username}\n\n")
        f.write("### Active User Profile (Top 10 Weights)\n")
        for tag, weight in top_tags:
            f.write(f"* **{tag}**: {weight:.4f}\n")
        f.write("\n")

        f.write("### Manual Rejections (0.35 Penalty Applied)\n")
        if newly_rejected_names:
            for name in newly_rejected_names:
                f.write(f"* ❌ User manually removed `{name}` from playlist.\n")
        else:
            f.write("* *No new manual rejections detected.*\n")
        f.write("\n")
        
        f.write("### Reversion Triggers\n")
        reversions_found = False
        for rec in validated_top_10:
            if "SwappedFrom" in rec:
                f.write(f"* Swapped sequel `{rec['SwappedFrom']}` for older entry `{rec['Item']['Name']}`\n")
                reversions_found = True
        if not reversions_found:
            f.write("* *No reversions triggered during this run.*\n")
        f.write("\n---\n\n")

def run_collision_scan(feature_matrix, all_items):
    """Scans the library for >95% similarity and logs it."""
    log_path = os.getenv("LOG_PATH", "recommender_log.md")
    sim_matrix = cosine_similarity(feature_matrix)
    
    collisions = []
    num_items = len(all_items)
    
    # Check upper triangle to avoid duplicate pairs and self-matches
    for i in range(num_items):
        for j in range(i + 1, num_items):
            if sim_matrix[i, j] > 0.95:
                collisions.append((all_items[i].get("Name"), all_items[j].get("Name"), sim_matrix[i, j]))
    
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("## Global Library Collision Alert (>95% Similarity)\n\n")
        if not collisions:
            f.write("* *No collisions detected.*\n")
        else:
            for item1, item2, score in collisions:
                f.write(f"* ⚠️ **{item1}** & **{item2}** (Similarity: {score:.2f})\n")
        f.write("\n---\n\n")

def apply_negative_feedback(user_id, base_user_vector, current_ticks, feature_matrix, item_id_to_index, manually_removed_ids, last_played_dict, id_to_name):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    TICKS_PER_HOUR = 36000000000 
    MAX_PENALTY_FACTOR = 0.20 
    REJECTION_PENALTY_FACTOR = 0.35 
    now = datetime.now(timezone.utc) 
    
    adjusted_vector = np.copy(base_user_vector)
    newly_rejected_names = []
    
    # 1. Process Definitive Manual Rejections First
    for item_id in manually_removed_ids:
        # Check the 24-hour watch buffer
        was_watched_recently = False
        last_played = last_played_dict.get(item_id)
        if last_played and (now - last_played).total_seconds() < 86400:
            was_watched_recently = True
        
        if not was_watched_recently:
            item_name = id_to_name.get(item_id, "Unknown Item")
            newly_rejected_names.append(item_name)
            cursor.execute('''
                UPDATE active_recommendations SET status = 'rejected' 
                WHERE user_id = ? AND item_id = ?
            ''', (user_id, item_id))

    # 2. Fetch all relevant items to apply mathematical penalties
    cursor.execute('''
        SELECT item_id, user_watch_ticks_at_rec, recommended_at, status FROM active_recommendations
        WHERE user_id = ? AND status IN ('pending', 'ignored', 'rejected')
    ''', (user_id,)) 
    
    recs = cursor.fetchall()
    
    for item_id, ticks_at_rec, rec_at_str, status in recs:
        rec_at_date = datetime.strptime(rec_at_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        
        # Apply Permanent Rejection Penalty
        if status == 'rejected':
            if item_id in item_id_to_index:
                item_vector = feature_matrix[item_id_to_index[item_id]]
                adjusted_vector -= (item_vector * REJECTION_PENALTY_FACTOR)
            continue 
            
        # The 6-Month Cooldown (Only for passive ignores)
        if (now - rec_at_date).days > 180:
            cursor.execute('''
                UPDATE active_recommendations SET status = 'cooldown' 
                WHERE user_id = ? AND item_id = ?
            ''', (user_id, item_id))
            continue
            
        # Verify it wasn't partially/fully watched recently
        cursor.execute('''
            SELECT completion_percentage FROM watch_history
            WHERE user_id = ? AND item_id = ?
        ''', (user_id, item_id))
        watch_data = cursor.fetchone()
        
        if watch_data and watch_data[0] > 0:
            cursor.execute('''
                UPDATE active_recommendations SET status = 'watched' 
                WHERE user_id = ? AND item_id = ?
            ''', (user_id, item_id))
            continue
            
        # The S-Curve Math (Active Hours)
        active_hours = (current_ticks - ticks_at_rec) / TICKS_PER_HOUR
        penalty = 0
        
        if active_hours >= 30:
            penalty = MAX_PENALTY_FACTOR
            if status == 'pending':
                cursor.execute('''
                    UPDATE active_recommendations SET status = 'ignored' 
                    WHERE user_id = ? AND item_id = ?
                ''', (user_id, item_id))
        elif active_hours >= 20:
            progress = (active_hours - 20) / 10.0
            penalty = MAX_PENALTY_FACTOR * (progress ** 2 * (3 - 2 * progress))
            
        if penalty > 0 and item_id in item_id_to_index:
            item_vector = feature_matrix[item_id_to_index[item_id]]
            adjusted_vector -= (item_vector * penalty)
                
    conn.commit()
    conn.close()
    
    return np.maximum(adjusted_vector, 0), newly_rejected_names

def process_user(user_id, username):
    print(f"\n--- Processing User: {username} ---")
    headers = {"Authorization": f"MediaBrowser Token={API_KEY}", "Content-Type": "application/json"}
    
    # --- TRACKING: WHAT DID WE RECOMMEND YESTERDAY? ---
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS expected_playlist (
            user_id TEXT,
            item_id TEXT,
            PRIMARY KEY (user_id, item_id)
        )
    ''')
    cursor.execute('SELECT item_id FROM expected_playlist WHERE user_id = ?', (user_id,))
    expected_playlist_ids = {row[0] for row in cursor.fetchall()}
    conn.close()
    # --------------------------------------------------

    # --- FETCH CURRENT PLAYLIST STATE ---
    playlist_name = f"Recommended for {username}"
    search_resp = requests.get(f"{SERVER_URL}/Users/{user_id}/Items", headers=headers, params={"SearchTerm": playlist_name, "IncludeItemTypes": "Playlist", "Recursive": "true"})
    existing_playlists = search_resp.json().get("Items", [])
    playlist_id = next((pl.get("Id") for pl in existing_playlists if pl.get("Name") == playlist_name), None)
    
    current_playlist_ids = set()
    if playlist_id:
        items_resp = requests.get(f"{SERVER_URL}/Playlists/{playlist_id}/Items", headers=headers, params={"UserId": user_id, "Fields": "SeriesId"})
        for i in items_resp.json().get("Items", []):
            current_playlist_ids.add(i.get("Id"))
            if i.get("SeriesId"): 
                current_playlist_ids.add(i.get("SeriesId"))
                
    # Calculate exact manual removals
    manually_removed_ids = set()
    if expected_playlist_ids:
        manually_removed_ids = expected_playlist_ids - current_playlist_ids
    # ------------------------------------

    params = {"IncludeItemTypes": "Movie,Series", "Recursive": "true", "Fields": "Tags,Genres,UserData,ProductionYear,SortName,People"}
    response = requests.get(f"{SERVER_URL}/Users/{user_id}/Items", headers=headers, params=params)
    all_items = response.json().get("Items", [])
    
    interacted_items = []
    candidate_items = []
    strictly_unwatched_items = []
    last_played_dict = {}
    id_to_name = {}
    now = datetime.now(timezone.utc)
    
    for item in all_items:
        item_id = item.get("Id")
        id_to_name[item_id] = item.get("Name")
        
        user_data = item.get("UserData", {})
        played = user_data.get("Played", False)
        play_count = user_data.get("PlayCount", 0)
        playback_ticks = user_data.get("PlaybackPositionTicks", 0)
        last_played_str = user_data.get("LastPlayedDate")
        
        if played or play_count > 0 or playback_ticks > 0:
            interacted_items.append(item)

        if not played:
            strictly_unwatched_items.append(item)
            
        is_candidate = True
        if last_played_str:
            try:
                last_played_date = datetime.strptime(last_played_str[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
                last_played_dict[item_id] = last_played_date
                if (now - last_played_date).days < 180:
                    is_candidate = False 
            except ValueError:
                pass
                
        if is_candidate:
            candidate_items.append(item)

    current_ticks = log_to_database(user_id, username, interacted_items)
    vectorizer, feature_matrix, item_id_to_index = build_feature_matrix(all_items)
    user_vector = get_user_preference_vector(user_id, feature_matrix, item_id_to_index)
    
    if user_vector is None:
        print(f"Skipping {username}: Not enough watch history.")
        return

    user_vector, newly_rejected_names = apply_negative_feedback(user_id, user_vector, current_ticks, feature_matrix, item_id_to_index, manually_removed_ids, last_played_dict, id_to_name)

    gradient_playlist = get_gradient_recommendations(user_vector, candidate_items, feature_matrix, item_id_to_index)
    validated_top_10 = apply_reversion_logic(gradient_playlist, strictly_unwatched_items)
    
    log_active_recommendations(user_id, validated_top_10, current_ticks)
    append_user_audit_log(username, user_vector, vectorizer, validated_top_10, newly_rejected_names)

    new_item_ids = []
    for rec in validated_top_10:
        item = rec["Item"]
        if item["Type"] == "Series":
            ep_params = {"UserId": user_id, "IsPlayed": "false", "IsMissing": "false", "IsVirtualUnaired": "false", "Limit": 1, "Fields": "Id,Path,LocationType", "SortBy": "PremiereDate,SortName", "SortOrder": "Ascending"}
            ep_resp = requests.get(f"{SERVER_URL}/Shows/{item['Id']}/Episodes", headers=headers, params=ep_params)
            episodes = ep_resp.json().get("Items", [])
            if episodes and episodes[0].get("Path") and episodes[0].get("LocationType") != "Virtual":
                new_item_ids.append(episodes[0]["Id"])
        else:
            new_item_ids.append(item["Id"])

    if not new_item_ids:
        print("No valid items found to add.")
        return

    if playlist_id:
        items_resp = requests.get(f"{SERVER_URL}/Playlists/{playlist_id}/Items", headers=headers, params={"UserId": user_id})
        entry_ids = [i.get("PlaylistItemId") for i in items_resp.json().get("Items", []) if i.get("PlaylistItemId")]
        if entry_ids:
            requests.delete(f"{SERVER_URL}/Playlists/{playlist_id}/Items", headers=headers, params={"entryIds": ",".join(entry_ids), "userId": user_id})
        requests.post(f"{SERVER_URL}/Playlists/{playlist_id}/Items", headers=headers, params={"ids": ",".join(new_item_ids), "userId": user_id})
        print(f"Success! Playlist '{playlist_name}' updated.")
    else:
        requests.post(f"{SERVER_URL}/Playlists", headers=headers, json={"Name": playlist_name, "Ids": new_item_ids, "UserId": user_id, "MediaType": "Video"})
        print(f"Success! Playlist '{playlist_name}' created.")

    # --- SAVE EXPECTED PLAYLIST FOR NEXT RUN ---
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM expected_playlist WHERE user_id = ?', (user_id,))
    for item_id in new_item_ids:
        cursor.execute('INSERT INTO expected_playlist (user_id, item_id) VALUES (?, ?)', (user_id, item_id))
    conn.commit()
    conn.close()
    # -------------------------------------------

if __name__ == "__main__":
    initialize_markdown_log()
    
    headers = {"Authorization": f"MediaBrowser Token={API_KEY}", "Content-Type": "application/json"}
    users = requests.get(f"{SERVER_URL}/Users", headers=headers).json()
    
    # Fetch all items using the first user to run the global collision scan
    if users:
        first_user_id = users[0].get("Id")
        params = {"IncludeItemTypes": "Movie,Series", "Recursive": "true", "Fields": "Tags,Genres,ProductionYear,People"}
        items_resp = requests.get(f"{SERVER_URL}/Users/{first_user_id}/Items", headers=headers, params=params)
        all_items = items_resp.json().get("Items", [])
        
        vectorizer, feature_matrix, _ = build_feature_matrix(all_items)
        run_collision_scan(feature_matrix, all_items)
    
    for user in users:
        process_user(user.get("Id"), user.get("Name"))
