import sqlite3
import os

db_name = 'exam_stress.db'


confirm = input(f"Are you sure you want to DELETE all data in {db_name}? (y/n): ")
if confirm.lower() == 'y':
    if os.path.exists(db_name):
        conn = sqlite3.connect(db_name)
        c = conn.cursor()
        c.execute('DROP TABLE IF EXISTS students')
        c.execute('''
            CREATE TABLE students (
                student_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                genre TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()
        print("Database successfully reset.")
    else:
        print("Database file not found.")
else:
    print("Reset cancelled.")