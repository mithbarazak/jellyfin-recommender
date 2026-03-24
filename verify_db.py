import sqlite3

# --- Configuration ---
DB_PATH = "/mnt/Plex/jellyfin-sqlite/watch_history.db"

def verify_database():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM watch_history")
        total_records = cursor.fetchone()[0]
        print(f"Total watch history records logged: {total_records}\n")
        
        if total_records > 0:
            print("Records per user:")
            cursor.execute("SELECT username, COUNT(*) FROM watch_history GROUP BY username")
            user_counts = cursor.fetchall()
            
            for row in user_counts:
                print(f"- {row[0]}: {row[1]} items logged")
                
        conn.close()
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")

if __name__ == "__main__":
    verify_database()
