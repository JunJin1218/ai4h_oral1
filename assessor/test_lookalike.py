"""
Test the assessor model on labeled lookalike pairs from annotation files.
"""
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

from assessor.model import load_assessor


def _stem_base(s: str) -> str:
    """Strip trailing _0, _1, _2 etc. from stem for matching."""
    return re.sub(r"_\d+$", "", s)


def find_embedding_path(image_name: str, data_dir: Path) -> Path | None:
    """
    Find the .pt file corresponding to an image filename.
    Searches (in order): data/embeddings/, data/query/, data/query images/, data/query_images/, data/.
    Handles:
    - Exact stem match (e.g. 4049.jpg -> 4049.pt)
    - With/without extension, case-insensitive
    - Trailing _0, _1 on embedding (e.g. "1 Acarbose tab front YSP" -> "1 Acarbose tab front YSP_0.pt")
    - Excel name without suffix matching embedding base (e.g. "Acarbose ..." matching "Acarbose ... _0.pt")
    """
    stem = Path(image_name).stem
    stem_lower = stem.lower()
    stem_base = _stem_base(stem)

    # Search order: embeddings (candidates), then query folder(s), then data root
    candidate_dirs = [
        data_dir / "embeddings",
        data_dir / "query",
        data_dir / "query images",
        data_dir / "query_images",
    ]
    search_dirs = [d for d in candidate_dirs if d.exists()]
    search_dirs.append(data_dir)

    def search_in(directory: Path) -> Path | None:
        if not directory.exists():
            return None
        # Exact match
        exact = directory / f"{stem}.pt"
        if exact.exists():
            return exact
        for pt_file in directory.rglob("*.pt"):
            pt_stem = pt_file.stem
            pt_stem_lower = pt_stem.lower()
            pt_base = _stem_base(pt_stem)
            if pt_stem_lower == stem_lower:
                return pt_file
            if pt_base.lower() == stem_lower or pt_stem_lower == stem_base.lower():
                return pt_file
            # Excel "1 Acarbose tab front YSP" vs embedding "1 Acarbose tab front YSP_0"
            if pt_base.lower() == stem_base.lower():
                return pt_file
        return None

    for directory in search_dirs:
        found = search_in(directory)
        if found is not None:
            return found

    # Case-insensitive and suffix variants in all dirs
    for directory in search_dirs:
        if not directory.exists():
            continue
        for pt_file in directory.rglob("*.pt"):
            if pt_file.stem.lower() == stem_lower:
                return pt_file
            if _stem_base(pt_file.stem).lower() == stem_lower:
                return pt_file
            # Try appending _0, _1 to stem (Excel name might lack suffix)
            for suffix in ("_0", "_1", "_2"):
                if (pt_file.stem.lower() == (stem + suffix).lower() or
                        pt_file.stem.lower() == stem_lower + suffix):
                    return pt_file

    return None


def load_annotations(excel_path: Path) -> list[tuple[str, str, int]]:
    """
    Load pairs from Excel file.
    Returns list of (query_image, candidate_image, label) where label=1 for lookalike, 0 for not.
    Handles multiple Excel formats found in the annotation files.
    """
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        print(f"Warning: Could not read {excel_path}: {e}")
        return []
    
    pairs = []
    filename = excel_path.name
    
    # Format 1: lookalike annotation.xlsx or lookalike_annotation.xlsx
    # Layout: one row per label type per query. E.g. row 2 = Lookalike candidates for query 1,
    #         row 3 = Non look-alike candidates for query 1; row 4/5 for query 2, etc.
    #         Query can be in col0 on first row only (col0 empty on row 3 reuses query from row 2).
    # Structure A: Col0=Query, Col1=Label ("Lookalike" / "Non look-alike"), Col2+=Candidates
    # Structure B: Col0=Query, Col2=Label, Col3+=Candidates
    # Structure C: Row 0 = headers "Lookalike" / "Non look-alike"; Row 1+ = candidates under those columns
    if "lookalike" in filename.lower() and "annotation" in filename.lower():
        current_query = None
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {filename}", leave=False):
            query_val = row.iloc[0] if len(row) > 0 else None
            if pd.notna(query_val) and str(query_val).strip():
                current_query = str(query_val).strip()
            
            if current_query is None:
                continue
            
            # Label may be in column 1 or column 2; candidates start after the label column
            label_val = None
            candidate_start_col = 3
            if len(row) > 1:
                v1 = row.iloc[1]
                s1 = str(v1).strip().lower() if pd.notna(v1) else ""
                if "look-alike" in s1 or "lookalike" in s1 or "non" in s1 or s1 in ("no", "n", "0"):
                    label_val = v1
                    candidate_start_col = 2
            if label_val is None and len(row) > 2:
                v2 = row.iloc[2]
                s2 = str(v2).strip().lower() if pd.notna(v2) else ""
                if "look-alike" in s2 or "lookalike" in s2 or "non" in s2 or s2 in ("no", "n", "0"):
                    label_val = v2
                    candidate_start_col = 3
            if pd.isna(label_val):
                continue
            
            label_str = str(label_val).strip().lower()
            if "look-alike" in label_str or "lookalike" in label_str:
                if "non" in label_str or "not" in label_str:
                    label = 0
                else:
                    label = 1
            elif (
                "non look-alike" in label_str
                or "non lookalike" in label_str
                or "non-lookalike" in label_str
                or "non look alike" in label_str
                or "not lookalike" in label_str
                or "not look-alike" in label_str
                or label_str in ("no", "n", "0")
            ):
                label = 0
            else:
                continue
            
            for col_idx in range(candidate_start_col, len(row)):
                candidate = row.iloc[col_idx]
                if pd.notna(candidate):
                    candidate_str = str(candidate).strip()
                    if candidate_str.lower() not in ['x', 'xx', 'nan', ''] and not candidate_str.startswith('UNCERTAIN'):
                        pairs.append((current_query, candidate_str, label))
        
        # Fallback: header-row format (row 0 = "Lookalike" / "Non look-alike" as column headers)
        if not pairs and len(df) >= 2:
            header_row = df.iloc[0]
            lookalike_cols = []
            non_lookalike_cols = []
            for j, cell in enumerate(header_row):
                if pd.isna(cell):
                    continue
                s = str(cell).strip().lower()
                if "non look-alike" in s or "non lookalike" in s or "non-lookalike" in s or "non look alike" in s or "not lookalike" in s:
                    non_lookalike_cols.append(j)
                elif "look-alike" in s or "lookalike" in s:
                    if "non" not in s and "not" not in s:
                        lookalike_cols.append(j)
            if lookalike_cols or non_lookalike_cols:
                for i in range(1, len(df)):
                    row = df.iloc[i]
                    query_val = row.iloc[0] if len(row) > 0 else None
                    query = str(query_val).strip() if pd.notna(query_val) and str(query_val).strip() else "query"
                    for j in lookalike_cols:
                        if j < len(row) and pd.notna(row.iloc[j]):
                            c = str(row.iloc[j]).strip()
                            if c and c.lower() not in ("x", "xx", "nan", ""):
                                pairs.append((query, c, 1))
                    for j in non_lookalike_cols:
                        if j < len(row) and pd.notna(row.iloc[j]):
                            c = str(row.iloc[j]).strip()
                            if c and c.lower() not in ("x", "xx", "nan", ""):
                                pairs.append((query, c, 0))
    
    # Format 2: query_set_annotation_CGH1.xlsx
    # Structure: Column 0 = Query, Column 1 = "look-alike" header, Columns 2+ = Lookalike candidates
    elif "query_set" in filename.lower():
        current_query = None
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {filename}", leave=False):
            query_val = row.iloc[0] if len(row) > 0 else None
            if pd.notna(query_val) and str(query_val).strip() and not str(query_val).strip().lower() == 'x':
                current_query = str(query_val).strip()
            
            if current_query is None:
                continue
            
            # All candidates in this file are lookalikes (label=1)
            for col_idx in range(2, len(row)):
                candidate = row.iloc[col_idx]
                if pd.notna(candidate):
                    candidate_str = str(candidate).strip()
                    # Stop at "x" markers
                    if candidate_str.lower() == 'x':
                        break
                    if candidate_str.lower() not in ['nan', '']:
                        pairs.append((current_query, candidate_str, 1))
    
    # Format 3: uncertain_negatives_annotation_CGH1.xlsx
    # Structure: Column 0 = Query, Columns contain "UNCERTAIN_NEGATIVES" candidates (label=0)
    elif "uncertain" in filename.lower() or "negative" in filename.lower():
        current_query = None
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {filename}", leave=False):
            query_val = row.iloc[0] if len(row) > 0 else None
            if pd.notna(query_val) and str(query_val).strip() and not str(query_val).strip().lower() == 'x':
                current_query = str(query_val).strip()
            
            if current_query is None:
                continue
            
            # Look for UNCERTAIN_NEGATIVES pattern or extract candidates
            for col_idx in range(1, len(row)):
                candidate = row.iloc[col_idx]
                if pd.notna(candidate):
                    candidate_str = str(candidate).strip()
                    # Stop at "x" or "XX" markers
                    if candidate_str.lower() in ['x', 'xx']:
                        break
                    # Extract filename from patterns like "UNCERTAIN_NEGATIVES: 667.jpg"
                    if "UNCERTAIN_NEGATIVES" in candidate_str:
                        # Extract the filename part
                        parts = candidate_str.split(":")
                        if len(parts) > 1:
                            candidate_str = parts[1].strip()
                    if candidate_str.lower() not in ['nan', ''] and not candidate_str.startswith('UNCERTAIN'):
                        pairs.append((current_query, candidate_str, 0))
    
    else:
        print(f"Unknown Excel format for {filename}, skipping...")
        return []

    return pairs


def load_annotations_streamlined(path: Path) -> list[tuple[str, str, int]]:
    """
    Load (query, candidate, label) from streamlined test-data Excel.

    Supported formats:
    1. Sheet-per-query: sheet name = main drug (query). Row 1 = label headers
       (e.g. "Lookalike", "Non lookalike"). Rows 2+ = candidate names under each column.
    2. Single sheet, lookalike-style: col0=query, col2=label text, col3+=candidates.
    3. Single sheet, 3-column: query, candidate, label (0/1 or Yes/No).
    CSV: query_image, candidate_image, label.
    """
    import csv as csv_module
    if path.suffix.lower() == ".csv":
        raw: list[tuple[str, str, int]] = []
        with open(path, newline="", encoding="utf-8") as f:
            r = csv_module.DictReader(f)
            for row in r:
                q = (row.get("query_image") or row.get("query") or "").strip()
                c = (row.get("candidate_image") or row.get("candidate") or "").strip()
                try:
                    label = int(row.get("label", row.get("human_label", "")))
                except (ValueError, TypeError):
                    continue
                if label not in (0, 1) or not q or not c:
                    continue
                raw.append((q, c, label))
        return raw

    try:
        xl = pd.ExcelFile(path)
    except Exception:
        return []

    pairs: list[tuple[str, str, int]] = []

    # Format: one sheet per query drug, row 0 = label headers, row 1+ = candidates
    for sheet_name in xl.sheet_names:
        df = pd.read_excel(xl, sheet_name=sheet_name, header=None)
        if df.empty or len(df) < 2:
            continue
        query = str(sheet_name).strip()
        if not query or query.lower() in ("nan", "sheet", "sheet1"):
            continue
        header_row = df.iloc[0]
        lookalike_cols: list[int] = []
        non_lookalike_cols: list[int] = []
        for j, cell in enumerate(header_row):
            if pd.isna(cell):
                continue
            s = str(cell).strip().lower()
            if "non look-alike" in s or "non lookalike" in s or "non look alike" in s or "non-lookalike" in s:
                non_lookalike_cols.append(j)
            elif "look-alike" in s or "lookalike" in s or "look alike" in s:
                lookalike_cols.append(j)
        for i in range(1, len(df)):
            row = df.iloc[i]
            for j in lookalike_cols:
                if j < len(row) and pd.notna(row.iloc[j]):
                    c = str(row.iloc[j]).strip()
                    if c and c.lower() not in ("x", "xx", "nan", ""):
                        pairs.append((query, c, 1))
            for j in non_lookalike_cols:
                if j < len(row) and pd.notna(row.iloc[j]):
                    c = str(row.iloc[j]).strip()
                    if c and c.lower() not in ("x", "xx", "nan", ""):
                        pairs.append((query, c, 0))

    if pairs:
        return pairs

    # Fallback: single sheet, lookalike-style or 3-column
    df = pd.read_excel(path, header=None)
    if df.empty:
        return []

    def parse_label(val) -> int | None:
        if pd.isna(val):
            return None
        s = str(val).strip().lower()
        if s in ("1", "yes", "y", "lookalike", "look-alike", "true"):
            return 1
        if s in ("0", "no", "n", "non lookalike", "non look-alike", "false"):
            return 0
        try:
            n = int(float(val))
            return n if n in (0, 1) else None
        except (ValueError, TypeError):
            return None

    start = 0
    if len(df) > 0:
        first_cell = str(df.iloc[0].iloc[0]).strip().lower() if pd.notna(df.iloc[0].iloc[0]) else ""
        if first_cell in ("query", "query image", "query_image", "query image name"):
            start = 1

    # Try lookalike-style: col0=query, col2=label text, col3+=candidates
    current_query = None
    for i in range(start, len(df)):
        row = df.iloc[i]
        query_val = row.iloc[0] if len(row) > 0 else None
        if pd.notna(query_val) and str(query_val).strip():
            current_query = str(query_val).strip()
        if current_query is None:
            continue
        label_val = row.iloc[2] if len(row) > 2 else None
        if pd.isna(label_val):
            continue
        label_str = str(label_val).strip().lower()
        if "look-alike" in label_str or "lookalike" in label_str:
            label = 1
        elif "non look-alike" in label_str or "non lookalike" in label_str:
            label = 0
        else:
            continue
        for col_idx in range(3, len(row)):
            candidate = row.iloc[col_idx]
            if pd.notna(candidate):
                candidate_str = str(candidate).strip()
                if candidate_str.lower() not in ("x", "xx", "nan", "") and not candidate_str.startswith("UNCERTAIN"):
                    pairs.append((current_query, candidate_str, label))
    if pairs:
        return pairs

    # Fallback: simple 3-column (query, candidate, label)
    if len(df.columns) >= 3:
        for i in range(start, len(df)):
            row = df.iloc[i]
            q = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
            c = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""
            label = parse_label(row.iloc[2] if len(row) > 2 else None)
            if label is not None and q and c:
                pairs.append((q, c, label))
    return pairs


def main():
    data_dir = Path("data")
    annotation_dir = data_dir / "Annotation"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model, cfg = load_assessor(device=device, in_dir="assessor/model")
    model.eval()
    
    # Load all pairs from annotation files
    all_pairs = []
    excel_files = [
        annotation_dir / "query_set_annotation_CGH1.xlsx",
        annotation_dir / "uncertain_negatives_annotation_CGH1.xlsx",
        annotation_dir / "lookalike annotation.xlsx",
    ]
    
    for excel_file in excel_files:
        if excel_file.exists():
            pairs = load_annotations(excel_file)
            print(f"Loaded {len(pairs)} pairs from {excel_file.name}")
            all_pairs.extend(pairs)
        else:
            print(f"Note: {excel_file.name} not found")
    
    if not all_pairs:
        print("\n⚠️  No pairs found in annotation files.")
        print("You may need to adjust column names in load_annotations() to match your Excel format.")
        print("\nFalling back to simple test on random pairs...")
        
        # Fallback: test on random pairs
        pt_files = sorted(data_dir.rglob("*.pt"))
        if len(pt_files) < 2:
            raise FileNotFoundError("Need at least 2 .pt files under data/")
        
        f1, f2 = pt_files[0], pt_files[1]
        e1 = torch.load(f1, map_location="cpu")
        e2 = torch.load(f2, map_location="cpu")
        
        e1 = e1.to(device)
        e2 = e2.to(device)
        
        with torch.no_grad():
            score = model(e1, e2)
        
        print(f"\nRandom pair test:")
        print(f"  {f1.name} vs {f2.name}")
        print(f"  Score: {float(score.item()):.4f} (1=lookalike, 0=not)")
        return
    
    # Test each pair
    y_true = []
    y_pred = []
    scores_list = []
    found_pairs = []
    missing_count = 0
    
    print(f"\nTesting {len(all_pairs)} pairs...")
    print("Progress will be shown below:\n")
    
    # Progress bar for testing pairs with periodic updates
    pbar = tqdm(all_pairs, desc="Testing pairs", unit="pair")
    for idx, (img1_name, img2_name, true_label) in enumerate(pbar):
        emb1_path = find_embedding_path(img1_name, data_dir)
        emb2_path = find_embedding_path(img2_name, data_dir)
        
        if emb1_path is None or emb2_path is None:
            missing_count += 1
            continue  # Skip if embeddings not found
        
        e1 = torch.load(emb1_path, map_location="cpu")
        e2 = torch.load(emb2_path, map_location="cpu")
        
        if not torch.is_tensor(e1) or not torch.is_tensor(e2):
            continue
        
        if e1.shape != (cfg.embedding_size,) or e2.shape != (cfg.embedding_size,):
            continue
        
        e1 = e1.to(device)
        e2 = e2.to(device)
        
        with torch.no_grad():
            score = model(e1, e2)
        
        pred_label = 1 if score.item() > 0.5 else 0
        
        y_true.append(true_label)
        y_pred.append(pred_label)
        scores_list.append(float(score.item()))
        found_pairs.append((img1_name, img2_name, true_label, pred_label, float(score.item()), 
                           str(emb1_path), str(emb2_path)))
        
        # Update progress bar with current stats every 500 pairs
        if (idx + 1) % 500 == 0 and len(y_true) > 0:
            current_acc = accuracy_score(y_true, y_pred)
            pbar.set_postfix({
                'Tested': len(y_true),
                'Accuracy': f'{current_acc:.3f}',
                'Missing': missing_count
            })
    
    print(f"\n✓ Completed testing!")
    print(f"  - Valid pairs tested: {len(y_true)}")
    print(f"  - Pairs skipped (missing embeddings): {missing_count}")
    if missing_count > 0:
        print(f"  - Success rate: {len(y_true)/(len(y_true)+missing_count)*100:.1f}% of pairs had matching embeddings")
    
    if not y_true:
        print("\n❌ No valid pairs found! Check that:")
        print("  1. Embeddings exist in data/ folder")
        print("  2. Image filenames in Excel match embedding filenames")
        return
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    # labels=[0,1] ensures 2x2 matrix even when only one class is present
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    print(f"\n{'='*60}")
    print(f"RESULTS: Tested {len(y_true)} pairs")
    print(f"{'='*60}")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Not  Lookalike")
    print(f"Actual Not     {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"      Lookalike {cm[1][0]:4d}  {cm[1][1]:4d}")
    
    # Show detailed examples with file paths
    print(f"\n{'='*60}")
    print("Detailed Examples (showing actual files used):")
    print(f"{'='*60}")
    
    # Show 2 examples: one correct, one incorrect (if available)
    correct_examples = [p for p in found_pairs if p[2] == p[3]]  # true_label == pred_label
    incorrect_examples = [p for p in found_pairs if p[2] != p[3]]  # true_label != pred_label
    
    examples_to_show = []
    if correct_examples:
        examples_to_show.append(("✓ CORRECT", correct_examples[0]))
    if incorrect_examples:
        examples_to_show.append(("✗ INCORRECT", incorrect_examples[0]))
    if len(examples_to_show) < 2 and found_pairs:
        # Fill with first few if needed
        for p in found_pairs[:2]:
            if p not in [ex[1] for ex in examples_to_show]:
                examples_to_show.append(("Sample", p))
                if len(examples_to_show) >= 2:
                    break
    
    for label, (img1_name, img2_name, true_lbl, pred_lbl, score_val, emb1_path, emb2_path) in examples_to_show:
        print(f"\n{label}")
        print(f"  Query Image Name:     {img1_name}")
        print(f"  Candidate Image Name:  {img2_name}")
        print(f"  Embedding File 1:      {emb1_path}")
        print(f"  Embedding File 2:      {emb2_path}")
        print(f"  True Label:           {true_lbl} ({'Lookalike' if true_lbl == 1 else 'Not Lookalike'})")
        print(f"  Predicted Label:      {pred_lbl} ({'Lookalike' if pred_lbl == 1 else 'Not Lookalike'})")
        print(f"  Model Score:          {score_val:.4f} (threshold: 0.5)")
    
    # Show summary of first 10 pairs (simpler format)
    print(f"\n{'='*60}")
    print("Summary of first 10 pairs:")
    print(f"{'='*60}")
    for img1, img2, true_lbl, pred_lbl, score_val, _, _ in found_pairs[:10]:
        status = "✓" if true_lbl == pred_lbl else "✗"
        print(f"{status} {img1[:40]:40s} vs {img2[:40]:40s} | "
              f"True: {true_lbl}, Pred: {pred_lbl}, Score: {score_val:.4f}")


if __name__ == "__main__":
    main()
