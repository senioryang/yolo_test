import os
import sys
import re
import json
from pathlib import Path
# import pymysql # Example: pip install pymysql
import requests # Used for downloading images
from sqlalchemy import create_engine, text

# Patch paramiko for compat with new versions removing top-level keys
import paramiko
try:
    if not hasattr(paramiko, "DSSKey"):
        from paramiko.dsskey import DSSKey
        paramiko.DSSKey = DSSKey
    if not hasattr(paramiko, "RSAKey"):
        from paramiko.rsakey import RSAKey
        paramiko.RSAKey = RSAKey
except (ImportError, AttributeError) as e:
    print(f"Warning: could not import paramiko keys directly: {e}")
    # Fallback for sshtunnel which requires these attributes to exist
    if not hasattr(paramiko, "DSSKey"):
        class DSSKey: pass
        paramiko.DSSKey = DSSKey
    if not hasattr(paramiko, "RSAKey"):
        class RSAKey: pass
        paramiko.RSAKey = RSAKey

import sshtunnel  # Add this import for SSH tunneling
import traceback # For detailed error info
from dotenv import load_dotenv

# Determine Project Root at module level for robust path resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# SSH and Database credentials
SSH_HOST = "101.200.164.23"
SSH_USER = "ecs-user"

# Resolve PEM path: If relative, assume it's in PROJECT_ROOT
_pem_env = os.getenv("PEM_FILE_PATH", "ecs-user-luti.pem")
if not os.path.isabs(_pem_env):
    SSH_PEM = str(PROJECT_ROOT / _pem_env)
else:
    SSH_PEM = _pem_env

print(f"PEM File resolved to: {SSH_PEM}")

IS_LOCAL = os.getenv("IS_LOCAL", "0") == "1"

MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DB = os.getenv("MYSQL_DB", "question_review")
# If local, use provided host. If using SSH tunnel, we will connect to localhost.
# The remote target for the tunnel is the RDS instance.
REMOTE_MYSQL_HOST = "rm-2ze7405gzw4zes7k4.mysql.rds.aliyuncs.com"
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))

def download_image(url, save_path):
    """
    Helper function to download image from URL to local path.
    """
    if not url: return
    
    try:
        # Check if file already exists
        if os.path.exists(save_path):
            print(f"Skipping existing file: {save_path.name}")
            return

        print(f"Downloading {url} to {save_path.name}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
            
    except Exception as e:
         print(f"Failed to establish SSH tunnel or connect to DB: {e}")
         traceback.print_exc()

def save_book_metadata(metadata, output_base_dir):
    """
    Saves or updates book metadata to a JSON file.
    Structure: { book_id: { 'subject': ..., 'degree': ... } }
    """
    meta_path = Path(output_base_dir) / "book_metadata.json"
    existing_data = {}
    
    if meta_path.exists():
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:
            pass
            
    # Update with new data
    existing_data.update(metadata)
    
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)
    print(f"Updated metadata for {len(metadata)} books in {meta_path.name}")

def fetch_data_by_book_id(book_id, output_base_dir):
    """
    Main function to fetch all necessary data for training from DB.
    """
    print(f"--- Starting Data Fetch for Book ID: {book_id} ---")
    
    # Establish SSH Tunnel and DB Connection
    try:
        with sshtunnel.SSHTunnelForwarder(
            (SSH_HOST, 22),
            ssh_username=SSH_USER,
            ssh_pkey=SSH_PEM,
            remote_bind_address=(REMOTE_MYSQL_HOST, 3306),
            local_bind_address=('127.0.0.1', 3306) # Use IPv4 localhost
        ) as tunnel:
            print(f"SSH Tunnel established. Local port: {tunnel.local_bind_port}")
            
            # Connect to the local end of the tunnel
            db_uri = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@127.0.0.1:{tunnel.local_bind_port}/{MYSQL_DB}"
            engine = create_engine(db_uri)
            
            # Use raw connection for backward compatibility with cursor logic
            conn = engine.raw_connection()
            
            if not conn:
                print("Failed to obtain database connection.")
                return

            # Directories setup matching YOLO project structure
            source_pages_dir = Path(output_base_dir) / "source_pages"
            source_crops_dir = Path(output_base_dir) / "source_crops"
            source_pages_dir.mkdir(parents=True, exist_ok=True)
            source_crops_dir.mkdir(parents=True, exist_ok=True)

            try:
                with conn.cursor() as cursor:
                    # [Pre-Step] Fetch book subject name and degree name
                    sql_info = f"""
                        SELECT 
                            bs.subject_name,
                            bd.degree_name
                        FROM books b
                        LEFT JOIN book_subjects bs ON b.subject_id = bs.subject_id
                        LEFT JOIN book_degrees bd ON b.degree_id = bd.degree_id
                        WHERE b.book_id = '{book_id}'
                        LIMIT 1
                    """
                    cursor.execute(sql_info)
                    info_row = cursor.fetchone()
                    
                    book_subject_name = 'UnknownSubject'
                    book_degree_name = 'UnknownDegree'

                    if info_row:
                        if isinstance(info_row, (tuple, list)):
                            if info_row[0]: book_subject_name = info_row[0]
                            if len(info_row) > 1 and info_row[1]: book_degree_name = info_row[1]
                        elif isinstance(info_row, dict):
                            book_subject_name = info_row.get('subject_name') or 'UnknownSubject'
                            book_degree_name = info_row.get('degree_name') or 'UnknownDegree'
                    
                    # Save Metadata for later stats analysis
                    save_book_metadata({
                        str(book_id): {
                            'subject': book_subject_name,
                            'degree': book_degree_name
                        }
                    }, output_base_dir)

                    # ---------------------------------------------------------
                    # Step 1: Query PDF Page Images
                    # Query the full page images for this book (the inputs for YOLO)
                    # ---------------------------------------------------------
                    print(f"Info: Subject={book_subject_name}, Degree={book_degree_name}")
                    print("Querying pages and creating directories...")
                    sql_pages = f"SELECT page_number, page_image_url FROM book_pages WHERE book_id = '{book_id}' ORDER BY page_number ASC"
                    cursor.execute(sql_pages)
                    pages = cursor.fetchall()
                    
                    print(f"Found {len(pages)} pages. Downloading sources...")
                    
                    for page in pages:
                        # Handle tuple/dict result safely
                        if isinstance(page, (tuple, list)):
                            p_num = int(page[0])
                            p_url = page[1]
                        else:
                            p_num = int(page.get('page_number'))
                            p_url = page.get('page_image_url')
                        
                        if not p_url: continue
                        
                        # UPDATE: Removed subject_name from filename
                        # New Naming: book_{book_id}_page_{p_num:04d}
                        base_name = f"book_{book_id}_page_{p_num:04d}"
                        
                        # 1. Download Page Image
                        # Determine extension
                        path_part = p_url.split('?')[0]  # Remove query params
                        ext = os.path.splitext(path_part)[1].lower()
                        if not ext or len(ext) > 5: ext = '.jpg'
                        
                        page_filename = f"{base_name}{ext}"
                        save_path = source_pages_dir / page_filename
                        
                        download_image(p_url, save_path)
                        
                        # 2. Create Crop Subdirectory (even for empty pages)
                        crop_subdir = source_crops_dir / base_name
                        crop_subdir.mkdir(parents=True, exist_ok=True)


                    # ---------------------------------------------------------
                    # Step 2: Query Structured Questions (Crops) & Page Ranges
                    # Query extracted question images and their logical page ranges
                    # ---------------------------------------------------------
                    print("Querying crops and ranges...")
                    
                    # Using raw SQL to match the logic of SQLAlchemy query in export_book_data
                    # Logic Steps:
                    # 1. Select questions by book_id
                    # 2. Join with latest review record (row_number window function)
                    # 3. Filter for correct reviews (is_correct = 1)
                    # 4. Sort by pdf_page_number and page_order
                    
                    sql_questions = f"""
                        WITH latest_reviews AS (
                            SELECT 
                                record_id,
                                question_id,
                                is_correct,
                                reviewed_question,
                                reviewed_option,
                                ROW_NUMBER() OVER (
                                    PARTITION BY question_id 
                                    ORDER BY reviewed_at DESC, record_id ASC
                                ) as row_num
                            FROM review_records
                            WHERE book_id = '{book_id}'
                        )
                        SELECT 
                            q.question_id,
                            q.pdf_page_number as start_page_num,
                            q.pdf_max_page_number as end_page_num,
                            r.reviewed_question,
                            r.reviewed_option
                        FROM questions q
                        JOIN latest_reviews r ON q.question_id = r.question_id
                        WHERE q.book_id = '{book_id}'
                          AND r.row_num = 1
                          AND r.is_correct = 1
                        ORDER BY q.pdf_page_number ASC, q.page_order ASC
                    """
                    
                    # Execute the query
                    cursor.execute(sql_questions)
                    questions = cursor.fetchall()
                    
                    print(f"Found {len(questions)} questions. Processing...")

                    for q in questions:
                        # Extract data safely with defaults
                        # Note: Dictionary access depends on cursor implementation. 
                        
                        # Use the pre-fetched subject name for all questions in this book
                        # subject_name = book_subject_name
                        
                        # Helper to get value whether dict or object or tuple (if we knew indices)
                        # But here we assume dict access as common in python db scripts
                        if isinstance(q, (tuple, list)):
                           # Fallback if tuple: id, start, end, q_text, o_text
                           q_id = q[0]
                           start_p = int(q[1]) if q[1] else 0
                           end_p = int(q[2]) if q[2] else start_p
                           q_text = q[3]
                           o_text = q[4]
                        else:
                           q_id = q.get('question_id')
                           start_p = int(q.get('start_page_num', 0))
                           end_p = int(q.get('end_page_num', 0)) or start_p
                           q_text = q.get('reviewed_question')
                           o_text = q.get('reviewed_option')
                        
                        # --- Create Directory Structure ---
                        # Use updated naming convention without subject
                        folder_name = f"book_{book_id}_page_{start_p:04d}"
                        save_folder = source_crops_dir / folder_name
                        save_folder.mkdir(parents=True, exist_ok=True)
                        
                        # --- Extract Image URLs ---
                        image_urls = []
                        
                        # 1. From Question Text (reviewed_question)
                        if q_text:
                            # Regex to find <img src="...">
                            found_in_q = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', str(q_text), re.IGNORECASE)
                            image_urls.extend(found_in_q)
                        
                        # 2. From Options (reviewed_option is JSON)
                        if o_text:
                            try:
                                # Try parsing as JSON first
                                options_dict = json.loads(o_text)
                                if isinstance(options_dict, dict):
                                    for key, val in options_dict.items():
                                        if isinstance(val, str):
                                            found_in_o = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', val, re.IGNORECASE)
                                            image_urls.extend(found_in_o)
                            except (json.JSONDecodeError, TypeError):
                                # Fallback: simple text regex search if not valid JSON
                                if isinstance(o_text, str):
                                     found_in_o_text = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', o_text, re.IGNORECASE)
                                     image_urls.extend(found_in_o_text)
                        
                        # --- Download Images ---
                        if not image_urls:
                            # print(f"  [Info] No images found for Question ID {q_id}")
                            continue
                            
                        for idx, url in enumerate(image_urls):
                            if not url: continue
                            # Construct usage-friendly filename: q{id}_p{start}-p{end}_{index}.{ext}
                            # Clean up URL to get extension
                            path_part = url.split('?')[0]  # Remove query params
                            ext = os.path.splitext(path_part)[1].lower()
                            if not ext or len(ext) > 5:
                                ext = '.jpg' # Default extension
                            
                            filename = f"q{q_id}_p{start_p}-p{end_p}_{idx}{ext}"
                            save_path = save_folder / filename
                            
                            download_image(url, save_path)
            
            except Exception as e:
                print(f"An error occurred during DB operations: {e}")
            finally:
                conn.close()
                print("--- Data Fetch Complete ---")
    except Exception as e:
         print(f"Failed to establish SSH tunnel or connect to DB: {e}")
         traceback.print_exc()

def get_all_book_ids():
    """
    Connects to DB and retrieves all available book IDs.
    """
    print("--- Fetching All Book IDs ---")
    ids = []
    # Use a Try/Finally block to ensure resources are cleaned up
    try:
        # Establish SSH Tunnel - similar to fetch_data_by_book_id
        # We use '0.0.0.0' or '127.0.0.1' for local binding.
        # Note: If running concurrent tunnels, ensure unique ports or reuse tunnel.
        tunnel = sshtunnel.SSHTunnelForwarder(
            (SSH_HOST, 22),
            ssh_username=SSH_USER,
            ssh_pkey=SSH_PEM,
            remote_bind_address=(REMOTE_MYSQL_HOST, 3306),
            local_bind_address=('127.0.0.1', 0)  # Use dynamic port (0) to avoid conflict
        )
        tunnel.start()
        
        try:
            print(f"SSH Tunnel established for ID fetch. Local port: {tunnel.local_bind_port}")
            db_uri = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@127.0.0.1:{tunnel.local_bind_port}/{MYSQL_DB}"
            engine = create_engine(db_uri)
            conn = engine.raw_connection()
            
            if conn:
                with conn.cursor() as cursor:
                    # Select ONLY book_ids
                    cursor.execute("SELECT book_id FROM books")
                    rows = cursor.fetchall()
                    for r in rows:
                        # Handle tuple vs dict depending on cursor
                        if isinstance(r, (list, tuple)):
                            ids.append(str(r[0]))
                        elif isinstance(r, dict):
                            ids.append(str(r.get('book_id')))
                    print(f"Successfully retrieved {len(ids)} book IDs.")
                conn.close()
        finally:
            tunnel.stop()
            
    except Exception as e:
        print(f"Error fetching book IDs: {e}")
        # traceback.print_exc()
    
    return ids

if __name__ == "__main__":
    # Configuration
    # Output path logic with SOLO_BASE_PATH support
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    # Reload .env from project root to be safe
    load_dotenv(PROJECT_ROOT / ".env")
    
    solo_base = os.getenv("SOLO_BASE_PATH")
    if solo_base:
        # User defined base path
        print(f"Using configured SOLO_BASE_PATH: {solo_base}")
        BASE_DATA_DIR = Path(solo_base) / "data"
    else:
        # Default project path
        BASE_DATA_DIR = PROJECT_ROOT / "data"
    
    print(f"Data directory set to: {BASE_DATA_DIR}")
    
    # Mode selection: specific book or all books
    # 1. Check for command line arguments
    book_ids_to_process = []
    if len(sys.argv) > 1:
        # User provided ID(s)
        book_ids_to_process = sys.argv[1:]
    else:
        # 2. Check for book_ids.txt in project root
        book_ids_file = PROJECT_ROOT / "book_ids.txt"
        if book_ids_file.exists():
            print(f"Loading book IDs from {book_ids_file}")
            try:
                with open(book_ids_file, "r", encoding="utf-8") as f:
                     content = f.read().strip()
                     if content:
                        # Split by newlines and filter empty lines
                        book_ids_to_process = [line.strip() for line in content.splitlines() if line.strip()]
                        print(f"Found {len(book_ids_to_process)} valid book IDs in file.")
                     else:
                        print("book_ids.txt is empty.")
            except Exception as e:
                print(f"Error reading book_ids.txt: {e}")
        else:
            print("book_ids.txt not found.")

    if not book_ids_to_process:
        print("No book IDs provided via arguments or book_ids.txt. Exiting.")
        sys.exit(0)
        
    print(f"Plan to process {len(book_ids_to_process)} books: {book_ids_to_process}")
    
    for b_id in book_ids_to_process:
        fetch_data_by_book_id(b_id, BASE_DATA_DIR)
