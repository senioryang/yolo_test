import os
import sys
import sshtunnel
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from pathlib import Path

# Determine Project Root at module level for robust path resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

SSH_HOST = "101.200.164.23"
SSH_USER = "ecs-user"
# Force correct PEM filename in path if not set from existing files
# Based on context
_pem_env = os.getenv("PEM_FILE_PATH", "ecs-user-luti.pem")
if not os.path.isabs(_pem_env):
    SSH_PEM = str(PROJECT_ROOT / _pem_env)
else:
    SSH_PEM = _pem_env

print(f"PEM File resolved to: {SSH_PEM}")
MYSQL_DB = os.getenv("MYSQL_DB", "question_review")
REMOTE_MYSQL_HOST = "rm-2ze7405gzw4zes7k4.mysql.rds.aliyuncs.com"
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))

# Patch paramiko for compat with new versions removing top-level keys
import paramiko
try:
    if not hasattr(paramiko, "DSSKey"):
        from paramiko.dsskey import DSSKey
        paramiko.DSSKey = DSSKey
    if not hasattr(paramiko, "RSAKey"):
        from paramiko.rsakey import RSAKey
        paramiko.RSAKey = RSAKey
except ImportError:
    pass

import traceback

def check_structure():
    print("Connecting to DB to check schema...")
    try:
        with sshtunnel.SSHTunnelForwarder(
            (SSH_HOST, 22),
            ssh_username=SSH_USER,
            ssh_pkey=SSH_PEM,
            remote_bind_address=(REMOTE_MYSQL_HOST, 3306),
            local_bind_address=('127.0.0.1', 0) 
        ) as tunnel:
            db_uri = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@127.0.0.1:{tunnel.local_bind_port}/{MYSQL_DB}"
            engine = create_engine(db_uri)
            
            with engine.connect() as conn:
                # 1. Inspect 'books' table columns
                print("\n[books] columns:")
                result = conn.execute(text("SHOW COLUMNS FROM books"))
                for row in result:
                    print(row)
                
                # 2. Check for any table with 'degree' in name
                print("\n[Tables] matching '%degree%':")
                result = conn.execute(text("SHOW TABLES LIKE '%degree%'"))
                degree_table = None
                for row in result:
                    print(row)
                    if 'degree' in str(row).lower():
                        degree_table = row[0]
                
                # 3. If degree table found, inspect columns
                if degree_table:
                    print(f"\n[{degree_table}] columns:")
                    result = conn.execute(text(f"SHOW COLUMNS FROM {degree_table}"))
                    for row in result:
                        print(row)

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    check_structure()
