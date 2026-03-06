import os
import csv
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

def generate_stats_report():
    # Paths
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    load_dotenv(PROJECT_ROOT / ".env")
    
    solo_base = os.getenv("SOLO_BASE_PATH")
    if solo_base:
        if isinstance(solo_base, str):
            solo_base = Path(solo_base)
        print(f"Using configured SOLO_BASE_PATH: {solo_base}")
        BASE_DIR = solo_base
    else:
        BASE_DIR = PROJECT_ROOT

    SOURCE_CROPS_DIR = BASE_DIR / "data" / "source_crops"
    SOURCE_PAGES_DIR = BASE_DIR / "data" / "source_pages" # Added Source Pages Dir
    LABELS_TRAIN_DIR = BASE_DIR / "dataset" / "labels" / "train"
    LABELS_VAL_DIR = BASE_DIR / "dataset" / "labels" / "val"
    OUTPUT_CSV = BASE_DIR / "runs" / "stats_report.csv"
    MATCH_LOG = BASE_DIR / "runs" / "matching_details.csv"
    METADATA_FILE = BASE_DIR / "data" / "book_metadata.json"
    
    # Ensure output directory exists
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    # Load Metadata
    book_metadata = {}
    if METADATA_FILE.exists():
        try:
            import json
            with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                book_metadata = json.load(f)
        except Exception:
            pass
            
    # Helper to get metadata
    def get_meta(page_or_book_str):
        # Extract book_id from "book_1234_page_..." or "book_1234"
        parts = page_or_book_str.split('_')
        if len(parts) >= 2 and parts[0] == 'book':
            bid = parts[1]
            if bid in book_metadata:
                return book_metadata[bid].get('degree', 'Unknown'), book_metadata[bid].get('subject', 'Unknown')
        return "Unknown", "Unknown"

    # Data structure: page_name -> {'original': 0, 'yolo': 0, 'conf_scores': []}
    stats = defaultdict(lambda: {'original': 0, 'yolo': 0, 'conf_scores': []})
    
    # 0. Load Matching Confidence Scores
    print("Loading matching confidence scores...")
    if MATCH_LOG.exists():
        with open(MATCH_LOG, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                page_name = row['page_name']
                try:
                    score = float(row['confidence_score'])
                    stats[page_name]['conf_scores'].append(score)
                except ValueError:
                    pass
    else:
        print(f"Warning: Match log not found at {MATCH_LOG}. Confidence scores will be empty.")

    # 1. Count Original Crops
    print("Counting original source crops...")
    if SOURCE_CROPS_DIR.exists():
        for page_dir in SOURCE_CROPS_DIR.iterdir():
            if page_dir.is_dir():
                page_name = page_dir.name
                # Count image files
                count = len([f for f in page_dir.glob("*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
                stats[page_name]['original'] = count
    else:
        print(f"Warning: Source crops directory not found at {SOURCE_CROPS_DIR}")

    # 2. Count YOLO Labels (Train + Val)
    print("Counting YOLO labels...")
    
    def count_labels_in_dir(label_dir):
        if not label_dir.exists():
            return
        for label_file in label_dir.glob("*.txt"):
            page_name = label_file.stem
            with open(label_file, 'r', encoding='utf-8') as f:
                # Count non-empty lines
                count = len([line for line in f if line.strip()])
                stats[page_name]['yolo'] = count

    count_labels_in_dir(LABELS_TRAIN_DIR)
    count_labels_in_dir(LABELS_VAL_DIR)
    
    # 3. Prepare Data for CSV
    # Sort by page name
    sorted_pages = sorted(stats.keys())
    
    csv_rows = []
    total_original = 0
    total_yolo = 0
    
    # Headers
    print("\n{:<40} | {:<12} | {:<12} | {:<12} | {:<8} | {:<8}".format("Page Name", "Degree", "Subject", "Orig Crops", "Yolo", "Diff"))
    print("-" * 120)
    
    for page in sorted_pages:
        orig = stats[page]['original']
        yolo = stats[page]['yolo']
        diff = yolo - orig
        degree, subject = get_meta(page)
        
        # Calculate scores
        scores = stats[page]['conf_scores']
        if scores:
            min_conf = f"{min(scores):.4f}"
            avg_conf = f"{sum(scores)/len(scores):.4f}"
        else:
            min_conf = "N/A"
            avg_conf = "N/A"
        
        if orig == 0 and yolo == 0:
            pass

        total_original += orig
        total_yolo += yolo
        
        csv_rows.append([page, degree, subject, orig, yolo, diff, min_conf, avg_conf])
        print("{:<40} | {:<12} | {:<12} | {:<12} | {:<8} | {:<8}".format(page, degree, subject, orig, yolo, diff))
        
    # Add Total Row
    total_diff = total_yolo - total_original
    csv_rows.append(["TOTAL", "-", "-", total_original, total_yolo, total_diff, "-", "-"])
    print("-" * 120)
    print("{:<40} | {:<12} | {:<12} | {:<12} | {:<8} | {:<8}".format("TOTAL", "-", "-", total_original, total_yolo, total_diff))

    # 4. Write CSV
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["Page Name", "Degree", "Subject", "Original Crops (Source)", "YOLO Labels (Generated)", "Difference", "Min Confidence", "Avg Confidence"])
        writer.writerows(csv_rows)
        
    print(f"\nStats report saved to: {OUTPUT_CSV}")

    # ==========================================
    # 5. Generate Book-level Report
    # ==========================================
    try:
        OUTPUT_BOOK_CSV = BASE_DIR / "runs" / "stats_book_report.csv"
        
        # 4.5. Count PDF Pages per Book
        print("\nCounting PDF pages per book...")
        book_page_counts = defaultdict(int)
        if SOURCE_PAGES_DIR and SOURCE_PAGES_DIR.exists():
            for f in SOURCE_PAGES_DIR.glob("*.*"):
                if f.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    continue
                # Parsing: "book_8785_page_0001.jpg" -> "book_8785"
                if "_page_" in f.stem:
                    book_key = f.stem.split("_page_")[0]
                    book_page_counts[book_key] += 1

        # Initialize Aggregator
        # book_key -> {'original': 0, 'yolo': 0, 'conf_scores': []}
        book_stats = defaultdict(lambda: {'original': 0, 'yolo': 0, 'conf_scores': []})

        print("\nAggregating book-level statistics...")
        for page_name, data in stats.items():
            # Parsing: "book_8785_page_0001" -> "book_8785"
            if "_page_" in page_name:
                book_key = page_name.split("_page_")[0]
            else:
                book_key = "unknown_book"

            book_stats[book_key]['original'] += data['original']
            book_stats[book_key]['yolo'] += data['yolo']
            book_stats[book_key]['conf_scores'].extend(data['conf_scores'])

        book_csv_rows = []
        # Ensure we cover all books found either via stats or via page counts
        all_book_keys = set(book_stats.keys()) | set(book_page_counts.keys())
        sorted_books = sorted(all_book_keys)

        print("\n" + "="*30 + " BOOK STATISTICS " + "="*30)
        # Header for console
        print("{:<30} | {:<12} | {:<12} | {:<12} | {:<12} | {:<8} | {:<8} | {:<8}".format("Book Name", "Degree", "Subject", "Total Pages", "Orig Crops", "Yolo Labels", "Diff", "Avg Conf"))
        print("-" * 140)

        for book in sorted_books:
            orig = book_stats[book]['original']
            yolo = book_stats[book]['yolo']
            diff = yolo - orig
            degree, subject = get_meta(book)
            total_pages = book_page_counts.get(book, 0)
            scores = book_stats[book]['conf_scores']
            
            if scores:
                min_conf = float(min(scores))
                avg_conf = float(sum(scores)/len(scores))
                s_min = f"{min_conf:.4f}"
                s_avg = f"{avg_conf:.4f}"
            else:
                s_min = "N/A"
                s_avg = "N/A"

            book_csv_rows.append([book, degree, subject, total_pages, orig, yolo, diff, s_min, s_avg])
            print("{:<30} | {:<12} | {:<12} | {:<12} | {:<12} | {:<8} | {:<8} | {:<8}".format(book, degree, subject, total_pages, orig, yolo, diff, s_avg))

        # Write Book CSV
        with open(OUTPUT_BOOK_CSV, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(["Book Name", "Degree", "Subject", "Total PDF Pages", "Original Crops", "YOLO Labels", "Difference", "Min Confidence", "Avg Confidence"])
            writer.writerows(book_csv_rows)
            
        print(f"\n[Book Level] Stats report saved to: {OUTPUT_BOOK_CSV}")

        print("(You can open this CSV file directly in Excel)")
    except Exception as e:
        print(f"\nERROR generating book stats: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_stats_report()
