import os
import sys
import re
import json
from pathlib import Path
# import pymysql # Example: pip install pymysql
import requests # Used for downloading images
from sqlalchemy import create_engine, text
import sshtunnel  # Add this import for SSH tunneling

from dotenv import load_dotenv
load_dotenv()


# SSH and Database credentials
SSH_HOST = "101.200.164.23"
SSH_USER = "ecs-user"
SSH_PEM = os.getenv("PEM_FILE_PATH", "ecs-user-luti.pem")

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
        print(f"Failed to download {url}: {e}")

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
                    # [Pre-Step] Fetch book subject name for naming
                    sql_subject = f"""
                        SELECT bs.subject_name 
                        FROM books b
                        LEFT JOIN book_subjects bs ON b.subject_id = bs.subject_id
                        WHERE b.book_id = '{book_id}'
                        LIMIT 1
                    """
                    cursor.execute(sql_subject)
                    subject_row = cursor.fetchone()
                    book_subject_name = 'Unknown'
                    if subject_row:
                        if isinstance(subject_row, (tuple, list)):
                            book_subject_name = subject_row[0]
                        elif isinstance(subject_row, dict):
                            book_subject_name = subject_row.get('subject_name')
                    if not book_subject_name: book_subject_name = 'Unknown'
                    
                    # ---------------------------------------------------------
                    # Step 1: Query PDF Page Images
                    # Query the full page images for this book (the inputs for YOLO)
                    # ---------------------------------------------------------
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
                        
                        # Naming Convention: book_{book_id}_{subject_name}_page_{page_num}
                        # Use zero-padding for correct sorting (0001, 0002...)
                        base_name = f"book_{book_id}_{book_subject_name}_page_{p_num:04d}"
                        
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
                        subject_name = book_subject_name
                        
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
                        
                        if not subject_name: subject_name = 'Unknown'
                        
                        # --- Create Directory Structure ---
                        # Requested format: book_{book_id}_{subject_name}_page_{start_page_num}
                        # Must match page filename logic with zero padding
                        folder_name = f"book_{book_id}_{subject_name}_page_{start_p:04d}"
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


if __name__ == "__main__":
    # Configuration
    BOOK_ID = "8785"
    
    # Output path: c:/ysc/github/yolo_test/data
    BASE_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
    
    fetch_data_by_book_id(BOOK_ID, BASE_DATA_DIR)
