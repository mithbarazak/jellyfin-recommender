# Jellyfin Smart Recommender

A custom, "vibe coded" (via Google Gemini) Python-based recommendation engine for a local Jellyfin media server. It runs as a daily batch script on a headless Ubuntu machine, providing CPU-efficient, multi-user adaptive recommendations using vector math and cosine similarity.

## Features

* **The "Super-Fingerprint":** Converts media items into a weighted vector space using Theme/Trope Tags, Decades, Genres, and People (Directors/Writers/Cast). It automatically deduplicates tags that mirror genres or decades to prevent artificial score inflation.
* **Dynamic User Profiles:** Calculates a rolling average of a user's tastes based on their actual watch history, tracked locally in SQLite.
* **4-3-3 Gradient Selection:** Prevents "filter bubbles" by slicing recommendations into pools: 4 "Safe" picks, 3 "Moderate" picks, and 3 "Reach" picks.
* **4-Tier Reversion Logic:** Ensures sequels are never recommended before unwatched prequels by scanning Franchise tags, Universe tags, DependsOn cross-references, and strict alphanumeric prefix sorting.
* **S-Curve Negative Feedback Loop:** Tracks how long a recommendation sits unwatched (using actual Jellyfin playback ticks, not wall-clock time). After 20 active hours, the engine applies an S-Curve penalty to the item's tags. After 30 active hours, the penalty maximizes. Penalties automatically clear after a 6-month cooldown.
* **Markdown Auditing:** Generates daily `.md` logs tracking global library metadata collisions (95%+ similarity), active user tag weights, and triggered reversion swaps.
* **Active Negative Feedback Tracking:** Monitors manual playlist removals to apply a permanent 35% preference penalty to rejected titles, ensuring the engine adapts to explicit user dislikes in real-time.
* **Smart Rejection Logic:** Compares current playlists against a persistent state of previously suggested items to accurately identify and penalize manual user rejections while avoiding false positives.

## Prerequisites

* Python 3.8+
* A Jellyfin Server and an API Key
* A Linux environment (Cron and Logrotate recommended for automation)

## Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/mithbarazak/jellyfin-recommender.git](https://github.com/mithbarazak/jellyfin-recommender.git)
   cd jellyfin-recommender
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Create a `.env` file in the root directory:
   ```text
   JELLYFIN_URL="http://YOUR_SERVER_IP:8096"
   JELLYFIN_API_KEY="your_api_key_here"
   DB_PATH="/path/to/your/storage/watch_history.db"
   LOG_PATH="/path/to/your/storage/recommender_log.md"
   ```

4. **Initialize the Database:**
   Run the setup script to generate the SQLite tables without affecting your media library:
   ```bash
   python setup_db_v2.py
   ```

5. **Run the Engine:**
   ```bash
   python recommend_items_v2.py
   ```

## Automation

It is recommended to run this script daily via a cron job during off-peak hours. Using the `nice` command ensures the script yields CPU priority to Jellyfin's transcoding tasks.

Example crontab entry (runs at 6:00 PM local time at the lowest CPU priority):
```text
0 18 * * * cd /path/to/jellyfin-recommender && nice -n 19 /path/to/jellyfin-recommender/venv/bin/python recommend_items_v2.py
```
