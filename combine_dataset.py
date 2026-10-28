import os
import glob
import pandas as pd

def merge_csv_files(input_folder, output_filepath):
    """
    Loads all CSV files from a folder and joins them into a single CSV file.
    Handles encoding errors automatically.
    """
    search_pattern = os.path.join(input_folder, "*.csv")
    all_files = glob.glob(search_pattern)
    
    if not all_files:
        print(f"No CSV files found in: {input_folder}")
        return

    print(f"Found {len(all_files)} CSV files to merge.")
    
    dataframes = []
    for file in all_files:
        file_name = os.path.basename(file)
        print(f"Reading: {file_name}")
        
        try:
            # Step 1: Try reading with standard UTF-8 first
            df = pd.read_csv(file, encoding="utf-8")
            dataframes.append(df)
        except UnicodeDecodeError:
            try:
                # Step 2: Fall back to cp1252 if UTF-8 fails
                print(f"  -> UTF-8 failed for {file_name}, trying cp1252...")
                df = pd.read_csv(file, encoding="cp1252")
                dataframes.append(df)
            except UnicodeDecodeError:
                # Step 3: Ultimate safety net. Force read by replacing bad characters.
                print(f"  -> cp1252 also failed for {file_name}. Forcing read with error replacement.")
                df = pd.read_csv(file, encoding="utf-8", on_bad_lines="skip", encoding_errors="replace")
                dataframes.append(df)
        except Exception as e:
            print(f"Unexpected error reading {file_name}: {e}")
            continue

    if not dataframes:
        print("No dataframes were successfully loaded. Aborting merge.")
        return

    print("Merging files...")
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Save output cleanly as standard UTF-8
    combined_df.to_csv(output_filepath, index=False, encoding="utf-8")
    print(f"\nSuccessfully saved merged file to: {output_filepath}")
    print(f"Total rows in merged dataset: {len(combined_df)}")


if __name__ == "__main__":
    INPUT_DIR = "./data/depression_dataset"                     
    OUTPUT_FILE = "./data/combined_tweets.csv"  
    
    merge_csv_files(INPUT_DIR, OUTPUT_FILE)