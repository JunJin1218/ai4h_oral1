import os
import re
import pandas as pd
from pathlib import Path

def extract_filenames_to_csv(directory_path: str, output_csv: str):
    """
    Recursively scans a directory for files and saves their names to a CSV.
    """
    dir_path = Path(directory_path)
    print(f"Looking inside: {dir_path.resolve()}")

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    # Extract just the file names
    filenames = [file.name for file in dir_path.rglob("*") if file.is_file()]
    print(f"Total files found: {len(filenames)}")

    # Save to CSV
    df = pd.DataFrame({"image_filename": filenames})
    df.to_csv(output_csv, index=False)
    print(f"Raw filenames exported to: {output_csv}")

def parse_medicine_filename(filename):
    """
    Parses a complex medicine filename into structured metadata components.
    """
    # Strip extension (.jpg, .png) and image sequence number (_0, _1)
    name_no_ext = re.sub(r'\.[^.]+$', '', filename)
    seq_match = re.search(r'_(\d+)$', name_no_ext)
    sequence = seq_match.group(1) if seq_match else ""
    base = re.sub(r'_(\d+)$', '', name_no_ext).strip()

    # Remove any stray image extensions hiding in the middle of the string
    base = re.sub(r'\.(jpg|jpeg|png|gif|bmp)', '', base, flags=re.IGNORECASE).strip()

    # Extract Date (in square brackets) and Version (e.g., v1, v2)
    date_match = re.search(r'\[(.*?)\]', base)
    date = date_match.group(1) if date_match else ""
    base = re.sub(r'\[.*?\]', '', base).strip()

    version_match = re.search(r'\sv(\d+)', base, re.IGNORECASE)
    version = version_match.group(1) if version_match else ""
    base = re.sub(r'\sv\d+', '', base, flags=re.IGNORECASE).strip()

    # Extract all parentheses
    parentheses = re.findall(r'\((.*?)\)', base)
    comp_part_outside = base
    for p in parentheses:
        comp_part_outside = comp_part_outside.replace(f'({p})', '')
    comp_part_outside = " ".join(comp_part_outside.split()).strip()

    # Process components and identify the composition parts
    ma_name = ""
    mfr_name = ""
    brand_name = ""
    container_type = ""
    generic_med_name = ""
    
    composition_blocks = []
    brand_container_blocks = []
    pkg_keywords = ['box', 'bottle', 'strip', 'front', 'rear', 'side', 'label', 'blister', 'top', 'bottom', 'back', 'pack', 'outer', 'carton']

    # Helper function to detect if a block contains chemical strengths
    def has_units(text):
        return bool(re.search(r'\d+(\.\d+)?\s*(mg|mcg|g|ml|%|iu|unit|u)\b', text, re.IGNORECASE))

    # Evaluate the text outside parentheses
    if has_units(comp_part_outside):
        composition_blocks.append(comp_part_outside)
    elif comp_part_outside:
        brand_container_blocks.append(comp_part_outside)

    for p in parentheses:
        # Check for tags and handle commas
        parts = [s.strip() for s in p.split(',')]
        is_metadata_paren = False
        current_tag = None
        
        for part in parts:
            p_low = part.lower()
            if p_low.startswith('ma&mfr_'):
                val = part[7:].strip()
                ma_name = val
                mfr_name = val
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
        
        if is_metadata_paren:
            continue
            
        # If it's not metadata, classify it dynamically based on units
        if has_units(p):
            composition_blocks.append(p)
        else:
            brand_container_blocks.append(p)

    # Compile the final components
    comp_part = " ".join(composition_blocks)
    
    # Safely extract brand and container
    for b in brand_container_blocks:
        b_low = b.lower()
        if any(k in b_low for k in pkg_keywords):
            if '-' in b:
                b_parts = b.split('-', 1)
                if not brand_name: brand_name = b_parts[0].strip()
                container_type = b_parts[1].strip()
            else:
                if not container_type: container_type = b
                else: container_type += f" {b}"
        else:
            if not brand_name: brand_name = b

    # Process the composition Part
    comp_part = re.sub(
        r'(mg|mcg|g|ml|%|iu|unit|u|l)\s*,\s*(\d+(\.\d+)?\s*(ml|mg|mcg|g|l|%|iu|unit|u)\b)', 
        r'\1/\2', 
        comp_part, 
        flags=re.IGNORECASE
    )

    raw_components = re.split(r'[,&+]', comp_part)
    chemicals = []
    strengths = []
    type_of_medicine = ""
    count_per_container = ""
    
    med_types = r'\b(effervescent tab|effervescent tablet|tab|soft cap|cap|capsule|tablet|gel|solution|oral susp|oral suspension|susp|suspension|cream|ointment|injection|vial|sachet|softgel|drops|syrup)\b'
    count_pattern = r'\b\d+[_]?[sS]\b'

    c_m = re.search(count_pattern, comp_part, re.IGNORECASE)
    if c_m: count_per_container = c_m.group(0)
    
    t_m = re.search(med_types, comp_part, re.IGNORECASE)
    if t_m: type_of_medicine = t_m.group(0)

    for rc in raw_components:
        rc = rc.strip()
        if not rc: continue
        
        rc_clean = rc
        if count_per_container: rc_clean = rc_clean.replace(count_per_container, "")
        if type_of_medicine: rc_clean = re.sub(rf'\b{type_of_medicine}\b', '', rc_clean, flags=re.IGNORECASE)
        rc_clean = rc_clean.strip()

        s_m = re.search(r'(\d+(\.\d+)?\s?(mg|mcg|g|ml|%|iu|unit|u)\b.*)', rc_clean, re.IGNORECASE)
        if s_m:
            strengths.append(s_m.group(1).strip())
            chemicals.append(rc_clean[:s_m.start()].strip())
        else:
            if rc_clean:
                chemicals.append(rc_clean)
                strengths.append("")

    # Generate a unique medicine ID for categorization
    core_components = []
    for i in range(len(chemicals)):
        core_components.append(chemicals[i])
        if i < len(strengths): core_components.append(strengths[i])
            
    core_components.append(brand_name if brand_name else generic_med_name)
    core_components.append(type_of_medicine)
    
    medicine_id = " | ".join([str(c).strip().lower() for c in core_components if str(c).strip() != ""])

    return {
        'Medicine_ID': medicine_id,
        'Original_Filename': filename,
        'Chemical_1': chemicals[0] if len(chemicals) > 0 else "",
        'Strength_1': strengths[0] if len(strengths) > 0 else "",
        'Chemical_2': chemicals[1] if len(chemicals) > 1 else "",
        'Strength_2': strengths[1] if len(strengths) > 1 else "",
        'Brand_Name': brand_name,
        'Container_Type': container_type,
        'Generic_Medication_Name': generic_med_name,
        'Type_of_Medicine': type_of_medicine,
        'Count_per_Container': count_per_container,
        'MA_Name': ma_name,
        'MFR_Name': mfr_name,
        'Version': version,
        'Date': date,
        'Image_Number': sequence
    }

if __name__ == "__main__":
    input_directory = "input_img"
    export_directory = "data/Metadata/"
    raw_csv_path = os.path.join(export_directory, "raw_filenames.csv")
    final_csv_path = os.path.join(export_directory, "medicine_metadata.csv")

    # Create the output directory if it doesn't exist
    if not os.path.exists(export_directory):
        os.makedirs(export_directory)
        print(f"Created new directory: {export_directory}")

    # Extract filenames to CSV
    try:
        extract_filenames_to_csv(input_directory, raw_csv_path)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the input directory exists.")
        exit(1)

    # Parse the raw filenames into structured metadata
    print("Loading raw_filenames.csv for metadata parsing...")
    try:
        df = pd.read_csv(raw_csv_path)
        print(f"Processing {len(df)} files into metadata...")
        
        parsed_data = [parse_medicine_filename(f) for f in df['image_filename']]
        df_final = pd.DataFrame(parsed_data)
        
        df_final.to_csv(final_csv_path, index=False)
        print(f"Data has been successfully exported to {final_csv_path}")
        
    except Exception as e:
        print(f"An error occurred during parsing: {e}")