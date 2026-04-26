"""
Filename Parser

Parses medication image filenames into structured metadata fields and exports the results to CSV.
The script first scans the input image directory, saves all filenames into a raw CSV, then applies
regex-based parsing rules to extract fields such as chemical name, strength, brand, container type,
manufacturer, date, image view type, and a generated Medicine_ID.

Usage:
    uv run python scripts/metadata_scripts/filename_parser.py

Outputs:
    - data/Metadata/raw_filenames.csv
    - data/Metadata/medicine_metadata.csv

Note:
    Update `input_directory` in the `__main__` block if your image folder changes.
"""

import os
import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm

pkg_keywords = ['box', 'bottle', 'strip', 'front', 'rear', 'side', 'label', 'blister', 'back', 'pack', 'outer', 'carton']

def extract_filenames_to_csv(directory_path: str, output_csv: str):
    """Recursively scans a directory for files and saves their names to a CSV."""
    dir_path = Path(directory_path)
    print(f"Looking inside: {dir_path.resolve()}")

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    filenames = [file.name for file in dir_path.rglob("*") if file.is_file()]
    print(f"Total files found: {len(filenames)}")

    df = pd.DataFrame({"image_filename": filenames})
    df.to_csv(output_csv, index=False)
    print(f"Raw filenames exported to: {output_csv}")

def parse_medicine_filename(filename):
    """Parses a complex medicine filename into structured metadata components."""
    name_no_ext = re.sub(r'\.[^.]+$', '', filename)
    name_no_ext = strip_stray_extension_tokens(name_no_ext)

    seq_match = re.search(r'_(\d+)$', name_no_ext)
    sequence = seq_match.group(1) if seq_match else ""

    base = re.sub(r'_(\d+)$', '', name_no_ext).strip()
    base, version = remove_standalone_version_tokens(base)

    # Medicine View Type Mapping
    view_type_mapping = {
        "0": "Front of Pill Blister pack",
        "1": "Back of Pill Blister pack",
        "2": "Front and Back of Pill Blister pack",
        "3": "Front of Box or Bottle",
        "4": "Back of Box or Bottle",
        "5": "Side 1 of Box or Bottle",
        "6": "Side 2 of Box or Bottle"
    }
    view_type = view_type_mapping.get(sequence, "")

    date_match = re.search(r'\[(.*?)\]', base)
    date = date_match.group(1) if date_match else ""
    base = re.sub(r'\[.*?\]', '', base).strip()

    parentheses = extract_top_level_parentheses(base)
    composition_paren_exists = any(is_composition_like_block(p) for p in parentheses)

    comp_part_outside = base
    for p in parentheses:
        comp_part_outside = comp_part_outside.replace(f'({p})', '', 1)

    comp_part_outside = " ".join(comp_part_outside.split()).strip()

    ma_name, mfr_name, brand_name, container_type, generic_med_name = "", "", "", "", ""
    composition_blocks = []
    brand_container_blocks = []

    def has_units(text):
        return bool(re.search(r'\d+(\.\d+)?\s*(mg|mcg|g|ml|%|iu|unit|u)\b', text, re.IGNORECASE))

    # If a clear composition already exists in parentheses, prefer that as the
    # chemistry source and treat outside text as the medicine name
    if composition_paren_exists:
        if comp_part_outside:
            brand_container_blocks.append(comp_part_outside)
    else:
        if has_units(comp_part_outside):
            composition_blocks.append(comp_part_outside)
        elif comp_part_outside:
            brand_container_blocks.append(comp_part_outside)

    for p in parentheses:
        parts = [s.strip() for s in p.split(',')]
        is_metadata_paren = False
        current_tag = None
        
        for part in parts:
            p_low = part.lower()
            if p_low.startswith('ma&mfr_'):
                val = part[7:].strip()
                ma_name, mfr_name = val, val
                current_tag = 'ma&mfr'
                is_metadata_paren = True
            elif p_low.startswith('ma_'):
                ma_name = part[3:].strip()
                current_tag = 'ma'
                is_metadata_paren = True
            elif p_low.startswith('mfr_'):
                mfr_name = part[4:].strip()
                current_tag = 'mfr'
                is_metadata_paren = True
            elif p_low.startswith('generic '):
                generic_med_name = part.strip()
                current_tag = 'generic'
                is_metadata_paren = True
            else:
                if current_tag == 'ma&mfr':
                    ma_name += ", " + part
                    mfr_name += ", " + part
                elif current_tag == 'ma':
                    ma_name += ", " + part
                elif current_tag == 'mfr':
                    mfr_name += ", " + part
                elif current_tag == 'generic':
                    generic_med_name += ", " + part
        
        if is_metadata_paren: continue
            
        if has_units(p):
            composition_blocks.append(p)
        elif looks_like_company_country_block(p):
            if not mfr_name:
                mfr_name = p.strip()
            else:
                mfr_name += ", " + p.strip()
        else:
            brand_container_blocks.append(p)

    comp_part = " ".join(composition_blocks)
    
    for b in brand_container_blocks:
        b = b.strip()
        if not b:
            continue

        b_low = b.lower()

        if any(k in b_low for k in pkg_keywords):
            if '-' in b:
                b_parts = b.split('-', 1)
                if not brand_name:
                    brand_name = b_parts[0].strip()
                container_type = b_parts[1].strip()
            else:
                if not container_type:
                    container_type = b
                else:
                    container_type += f" {b}"
        else:
            if not brand_name:
                brand_name = b

    comp_part = re.sub(
        r'(mg|mcg|g|ml|%|iu|unit|u|l)\s*,\s*(\d+(\.\d+)?\s*(ml|mg|mcg|g|l|%|iu|unit|u)\b)', 
        r'\1/\2', comp_part, flags=re.IGNORECASE
    )

    # Convert underscore between strength and next chemical into a normal separator
    comp_part = re.sub(
        r'(\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|%|iu|unit|u|l))_([A-Za-z])',
        r'\1 + \2',
        comp_part,
        flags=re.IGNORECASE
    )

    comp_part = expand_multi_chemical_shared_strengths(comp_part)
    comp_part = normalize_single_strength_format(comp_part)
    comp_part = normalize_strength_in_to_slash(comp_part)

    raw_components = re.split(r'[,&+]', comp_part)
    chemicals, strengths = [], []
    type_of_medicine, count_per_container = "", ""
    
    med_types = r'\b(effervescent tab|effervescent tablet|tab|soft cap|cap|capsule|tablet|gel|solution|soln|oral susp|oral suspension|susp|suspension|cream|ointment|injection|vial|sachet|softgel|drops|syrup|linctus)\b'
    count_pattern = r'\b\d+[_]?[sS]\b'

    c_m = re.search(count_pattern, comp_part, re.IGNORECASE)
    if c_m: count_per_container = c_m.group(0)
    
    t_m = re.search(med_types, comp_part, re.IGNORECASE)
    if not t_m:
        t_m = re.search(med_types, comp_part_outside, re.IGNORECASE)
    if t_m:
        type_of_medicine = normalize_medicine_type(t_m.group(0))

    for rc in raw_components:
        rc = rc.strip()
        if not rc: continue
        
        rc_clean = rc
        if count_per_container: rc_clean = rc_clean.replace(count_per_container, "")
        if type_of_medicine: rc_clean = re.sub(rf'\b{type_of_medicine}\b', '', rc_clean, flags=re.IGNORECASE)
        rc_clean = rc_clean.strip()

        s_m = re.search(
            r'(\d+(?:\.\d+)?\s?(mg|mcg|g|ml|%|iu|unit|u)\b.*)',
            rc_clean,
            re.IGNORECASE
        )

        if s_m:
            chem_part = rc_clean[:s_m.start()].strip()
            strength_part = s_m.group(1).strip()

            # Only create a new chemical-strength pair if a chemical is present
            if chem_part:
                chemicals.append(chem_part)
                strengths.append(strength_part)
            else:
                if strengths:
                    strengths[-1] = f"{strengths[-1]}{strength_part}"
        else:
            if rc_clean:
                chemicals.append(rc_clean)
                strengths.append("")
            
    # Fallback: if no chemical was found at all, treat the outside text as a generic name
    if not any(c.strip() for c in chemicals) and comp_part_outside:
        fallback_name = comp_part_outside

        if count_per_container:
            fallback_name = re.sub(
                rf'\b{re.escape(count_per_container)}\b', '', fallback_name, flags=re.IGNORECASE
            )

        if count_per_container:
            fallback_name = re.sub(
                rf'\b{re.escape(count_per_container)}_', '', fallback_name, flags=re.IGNORECASE
            )

        if type_of_medicine:
            fallback_name = re.sub(
                rf'\b{re.escape(type_of_medicine)}\b', '', fallback_name, flags=re.IGNORECASE
            )

        fallback_name = fallback_name.replace('_', ' ')
        fallback_name = " ".join(fallback_name.split()).strip(" -_,")

        if fallback_name:
            generic_med_name = fallback_name
            brand_name = ""

    # If composition came from parentheses, prefer the outside title text
    # before the dosage form as the generic name.
    if composition_paren_exists and comp_part_outside:
        extracted_name = extract_name_before_med_type(comp_part_outside, type_of_medicine)
        extracted_name = " ".join(extracted_name.split()).strip(" -_,")

        if extracted_name:
            generic_med_name = extracted_name
            brand_name = ""

    core_components = []
    for i in range(len(chemicals)):
        core_components.append(chemicals[i])
        if i < len(strengths): core_components.append(strengths[i])
            
    brand_for_id = normalize_brand_for_id(brand_name, type_of_medicine)
    name_for_id = brand_for_id if brand_for_id else generic_med_name

    core_components.append(name_for_id)
    core_components.append(type_of_medicine)

    medicine_id = " | ".join(
        [normalize_whitespace(str(c).lower()) for c in core_components if str(c).strip() != ""]
    )

    return {
        'Medicine_ID': medicine_id,
        'Original_Filename': filename,
        'Chemical_1': chemicals[0] if len(chemicals) > 0 else "",
        'Strength_1': strengths[0] if len(strengths) > 0 else "",
        'Chemical_2': chemicals[1] if len(chemicals) > 1 else "",
        'Strength_2': strengths[1] if len(strengths) > 1 else "",
        'Chemical_3': chemicals[2] if len(chemicals) > 2 else "",
        'Strength_3': strengths[2] if len(strengths) > 2 else "",
        'Brand_Name': brand_name,
        'Parsed_Container_Type': container_type,
        'Generic_or_Broad_Medicine_Name': generic_med_name,
        'Type_of_Medicine': type_of_medicine,
        'Count_per_Container': count_per_container,
        'MA_Name': ma_name,
        'MFR_Name': mfr_name,
        'Version': version,
        'Date': date,
        'Image_Number': sequence,
        'Medicine_View_Type': view_type
    }

def remove_standalone_version_tokens(text: str):
    """
    Removes version tokens like v1, v2 anywhere in the string.
    Treats spaces, underscores, hyphens, and punctuation as boundaries.
    """
    version = ""

    def repl(match):
        nonlocal version
        if not version:
            version = match.group(1)
        return ""

    cleaned = re.sub(
        r'(?i)(?<![A-Za-z0-9])v(\d+)(?![A-Za-z0-9])',
        repl,
        text
    )

    # Clean up separators left behind after removing version tokens
    cleaned = re.sub(r' {2,}', ' ', cleaned)
    cleaned = re.sub(r'_{2,}', '_', cleaned)
    cleaned = re.sub(r'-{2,}', '-', cleaned)
    cleaned = re.sub(r'_([)\]])', r'\1', cleaned)
    cleaned = re.sub(r'([(\[])\s+', r'\1', cleaned)
    cleaned = re.sub(r'\s+([)\]])', r'\1', cleaned)
    cleaned = cleaned.strip(" _-")

    return cleaned, version

def extract_top_level_parentheses(text: str):
    """
    Extracts balanced top-level parenthesized blocks, including nested parentheses.
    Returns a list of strings without the outer parentheses.
    """
    blocks = []
    start = None
    depth = 0

    for i, ch in enumerate(text):
        if ch == '(':
            if depth == 0:
                start = i
            depth += 1
        elif ch == ')':
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    blocks.append(text[start + 1:i])
                    start = None

    return blocks

def normalize_brand_for_id(brand_name: str, type_of_medicine: str) -> str:
    """
    Removes duplicated dosage-form words from the brand when building Medicine_ID.
    Example:
        brand_name='Anarex tab', type_of_medicine='Tab'
        -> 'Anarex'
    """
    brand = (brand_name or "").strip()
    med_type = (type_of_medicine or "").strip()

    if not brand:
        return ""

    if med_type:
        pattern = rf'(?i)\b{re.escape(med_type)}\b'
        brand = re.sub(pattern, '', brand).strip()

    brand = re.sub(r'\s{2,}', ' ', brand).strip()
    return brand

def is_composition_like_block(text: str) -> bool:
    """
    True if a block looks like an ingredient/composition block rather than
    metadata, packaging, or plain pack-size text.
    """
    text_low = text.lower().strip()

    # Exclude obvious metadata / packaging blocks
    if text_low.startswith(("ma_", "mfr_", "ma&mfr_", "generic ")):
        return False

    if any(k in text_low for k in pkg_keywords):
        return False

    # Exclude plain pack-volume blocks like "60ml", "100 ml", "1l"
    if re.fullmatch(r'\d+(\.\d+)?\s*(ml|l)\b', text_low, re.IGNORECASE):
        return False

    # Composition blocks usually contain active-strength patterns
    return bool(re.search(r'\d+(\.\d+)?\s*(mg|mcg|g|%|iu|unit|u)\b', text, re.IGNORECASE))

def extract_name_before_med_type(text: str, type_of_medicine: str) -> str:
    """
    Extract the full medicine text before the dosage form.
    Examples:
        'Dovato 50,300mg Tab' -> 'Dovato 50,300mg'
        'Entresto 50mg Tab' -> 'Entresto 50mg'
    """
    text = " ".join(text.split()).strip()
    med_type = (type_of_medicine or "").strip()

    if not text:
        return ""

    if med_type:
        m = re.search(rf'(?i)\b{re.escape(med_type)}\b', text)
        if m:
            return text[:m.start()].strip(" ,-_")

    return text.strip(" ,-_")

def looks_like_company_country_block(text: str) -> bool:
    """
    Heuristic for blocks like 'Novartis, Switzerland' or 'GSK, UK'.
    """
    t = text.strip()

    if not t:
        return False

    # Exclude explicit tagged metadata
    t_low = t.lower()
    if t_low.startswith(("ma_", "mfr_", "ma&mfr_", "generic ")):
        return False

    if any(k in t_low for k in pkg_keywords):
        return False

    # Exclude clear composition-like blocks
    if re.search(r'\d+(\.\d+)?\s*(mg|mcg|g|ml|%|iu|unit|u)\b', t, re.IGNORECASE):
        return False

    # Typical free-text manufacturer/country block has a comma
    return ',' in t

def strip_stray_extension_tokens(text: str) -> str:
    """
    Removes stray image-related tokens accidentally left inside the filename body,
    such as jpg, jpeg, png, gif, bmp, webp, tif, tiff, picture, pic, image, photo.

    Also removes an optional leading dot if present, e.g. '.jpg'.
    Only removes them when they appear as standalone tokens.
    """
    text = re.sub(
        r'(?i)(?<![A-Za-z0-9])\.?(jpg|jpeg|png|gif|bmp|webp|tif|tiff|picture|pic|image|photo)(?![A-Za-z0-9])',
        '',
        text
    )

    # Clean up leftover separators / spaces
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'_{2,}', '_', text)
    text = re.sub(r'-{2,}', '-', text)
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    text = re.sub(r'([ _-]+)([)\]])', r'\2', text)
    text = re.sub(r'([(\[])([ _-]+)', r'\1', text)
    text = text.strip(" ._-")
    return text

def expand_multi_chemical_shared_strengths(text: str) -> str:
    """
    Expands patterns like:
        Amlodipine,valsartan 5,160mg Tab
    into:
        Amlodipine 5mg + valsartan 160mg Tab

    Only applies when:
    - there are multiple comma-separated chemical names before the strength
    - the strength part is a comma-separated list of numbers followed by one unit
    - the number of chemicals matches the number of strengths
    """
    text = " ".join(text.split()).strip()

    m = re.match(
        r'^(?P<names>[A-Za-z][A-Za-z0-9\s\-]*(?:,[A-Za-z][A-Za-z0-9\s\-]*)+)\s+'
        r'(?P<vals>\d+(?:\.\d+)?(?:,\d+(?:\.\d+)?)+)'
        r'(?P<unit>mg|mcg|g|ml|%|iu|unit|u)\b'
        r'(?P<rest>.*)$',
        text,
        re.IGNORECASE
    )

    if not m:
        return text

    names = [n.strip() for n in m.group("names").split(",")]
    vals = [v.strip() for v in m.group("vals").split(",")]
    unit = m.group("unit")
    rest = m.group("rest").strip()

    if len(names) != len(vals):
        return text

    expanded = " + ".join(f"{n} {v}{unit}" for n, v in zip(names, vals))
    if rest:
        expanded += f" {rest}"

    return expanded

def normalize_single_strength_format(text: str) -> str:
    """
    Normalizes single-strength expressions without touching multi-chemical
    shared-strength patterns like 'Amlodipine,valsartan 5,160mg'.

    Examples:
        1,000mg      -> 1000mg
        100,000U,ml  -> 100000U/ml
        100,000IU,ml -> 100000IU/ml
    """
    # Remove commas used as thousands separators when directly before a unit
    text = re.sub(
        r'(?<=\d),(?=\d{3}\s*(mg|mcg|g|ml|iu|unit|u)\b)',
        '',
        text,
        flags=re.IGNORECASE
    )

    # Convert unit,ml style to unit/ml
    text = re.sub(
        r'(?i)\b(iu|unit|u)\s*,\s*(ml|l)\b',
        r'\1/\2',
        text
    )

    return text

def normalize_strength_in_to_slash(text: str) -> str:
    """
    Normalizes strength expressions like:
        250mg in 5ml -> 250mg/5ml
        15mg in 5ml  -> 15mg/5ml
    """
    return re.sub(
        r'(?i)(\d+(?:\.\d+)?\s*(?:mg|mcg|g|iu|unit|u|%))\s+in\s+(\d+(?:\.\d+)?\s*(?:ml|l))',
        r'\1/\2',
        text
    )

def normalize_medicine_type(med_type: str) -> str:
    """
    Standardizes medicine type labels to canonical names.
    """
    if not med_type:
        return ""

    med_type_clean = med_type.strip().lower()

    type_map = {
        "tab": "tablet",
        "tablet": "tablet",
        "cap": "capsule",
        "capsule": "capsule",
        "susp": "suspension",
        "suspension": "suspension",
        "oral susp": "oral suspension",
        "oral suspension": "oral suspension",
        "soln": "solution",
        "solution": "solution",
        "effervescent tab": "effervescent tablet",
        "effervescent tablet": "effervescent tablet",
        "soft cap": "soft capsule",
        "injection": "injection",
        "vial": "vial",
        "sachet": "sachet",
        "drops": "drops",
        "syrup": "syrup",
        "linctus": "linctus",
    }

    return type_map.get(med_type_clean, med_type_clean)

def normalize_whitespace(text: str) -> str:
    """
    Collapses repeated whitespace into a single space and trims ends.
    """
    return re.sub(r'\s+', ' ', text).strip()

if __name__ == "__main__":
    input_directory = "input_img/cleaned_dataset" # Change this to your input directory containing the updated cleaned dataset
    export_directory = "data/Metadata/"
    raw_csv_path = os.path.join(export_directory, "raw_filenames.csv")
    final_csv_path = os.path.join(export_directory, "medicine_metadata.csv")

    if not os.path.exists(export_directory):
        os.makedirs(export_directory)
        print(f"Created new directory: {export_directory}")

    # Extract filenames
    try:
        extract_filenames_to_csv(input_directory, raw_csv_path)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the input directory exists.")
        exit(1)

    # Parse filenames using regex
    try:
        df_raw = pd.read_csv(raw_csv_path)
        print(f"Processing {len(df_raw)} files through regex parsing...")
        
        final_data = []
        
        for filename in tqdm(df_raw['image_filename'], desc="Parsing Filenames", unit="img"):
            parsed_metadata = parse_medicine_filename(filename)
            final_data.append(parsed_metadata)

        # Save to CSV
        df_final = pd.DataFrame(final_data)
        
        df_final.to_csv(final_csv_path, index=False)
        print(f"\nSuccess! Parsed metadata has been exported to {final_csv_path}")
        
    except Exception as e:
        print(f"An error occurred during execution: {e}")