"""
Reset the database by dropping and recreating all tables.
WARNING: This will delete all data!
"""
import sys
sys.path.insert(0, '.')

from app.db.migrations import drop_all_tables, init_db

if __name__ == "__main__":
    print("=" * 70)
    print("  RESETTING DATABASE")
    print("=" * 70)
    print()
    print("⚠️  WARNING: This will delete all data!")
    print()
    
    try:
        drop_all_tables()
        init_db()
        print()
        print("=" * 70)
        print("  DATABASE RESET SUCCESSFULLY")
        print("=" * 70)
        print()
        print("✅ Database tables have been recreated")
        print("   You can now upload documents")
    except Exception as e:
        print()
        print(f"❌ Error resetting database: {e}")
        sys.exit(1)
