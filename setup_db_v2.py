import sqlite3
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
DB_PATH = os.getenv("DB_PATH")

def upgrade_database():
    if not DB_PATH:
        print("Error: DB_PATH not found in .env file.")
        return

    print(f"Connecting to database at: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Table 1: Store the dynamic user preference vector
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id TEXT PRIMARY KEY,
            username TEXT,
            preference_vector TEXT, 
            total_watch_ticks INTEGER DEFAULT 0,
            last_updated TEXT
        )
    ''')
    
    # Table 2: Track active recommendations for the 30-hour negative feedback loop
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS active_recommendations (
            user_id TEXT,
            item_id TEXT,
            recommended_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            user_watch_ticks_at_rec INTEGER,
            status TEXT DEFAULT 'pending',
            UNIQUE(user_id, item_id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Success: Database schema updated. Your existing watch history was not altered.")

if __name__ == "__main__":
    upgrade_database()
