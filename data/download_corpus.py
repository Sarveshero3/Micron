import os
import zipfile
import urllib.request
import sys

def download_and_extract():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    zip_path = os.path.join(script_dir, "cornell_movie_dialogs_corpus.zip")
    extract_dir = os.path.join(script_dir, "cornell_movie_dialogs_corpus")
    output_file = os.path.join(script_dir, "qa_dataset.txt")
    
    url = "http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"
    
    print(f"Downloading Cornell Movie-Dialogs Corpus from:\n{url}")
    try:
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete!")
    except Exception as e:
        print(f"Error downloading: {e}")
        print("Trying fallback mirror URL...")
        # fallback mirror
        fallback_url = "https://zissou.infosci.cornell.edu/socialmedia/playlists/cornell_movie_dialogs_corpus.zip"
        try:
            urllib.request.urlretrieve(fallback_url, zip_path)
            print("Download complete from mirror!")
        except Exception as err:
            print(f"Fallback download failed: {err}")
            sys.exit(1)
            
    print("Extracting ZIP file...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(script_dir)
        print("Extraction complete!")
    except Exception as e:
        print(f"Error unzipping: {e}")
        sys.exit(1)
        
    # Clean up zip
    if os.path.exists(zip_path):
        os.remove(zip_path)
        
    print("Processing dialogue files...")
    # Find the lines file and conversations file
    lines_file = None
    convs_file = None
    
    for root, dirs, files in os.walk(script_dir):
        for f in files:
            if f == "movie_lines.txt":
                lines_file = os.path.join(root, f)
            elif f == "movie_conversations.txt":
                convs_file = os.path.join(root, f)
                
    if not lines_file or not convs_file:
        print("Error: Could not find movie_lines.txt or movie_conversations.txt in the extracted directory.")
        sys.exit(1)
        
    # Load movie lines (Key: LineID, Value: Text)
    print("Loading movie lines...")
    lines = {}
    with open(lines_file, 'r', encoding='iso-8859-1') as f:
        for line in f:
            parts = line.split(" +++$+++ ")
            if len(parts) >= 5:
                line_id = parts[0]
                text = parts[4].strip()
                lines[line_id] = text
                
    print(f"Loaded {len(lines)} movie lines.")
    
    # Load conversations and compile Q&As
    print("Compiling conversation QA pairs...")
    qa_count = 0
    with open(convs_file, 'r', encoding='iso-8859-1') as f, open(output_file, 'w', encoding='utf-8') as out:
        for line in f:
            parts = line.split(" +++$+++ ")
            if len(parts) >= 4:
                # Extract line IDs list from string representation e.g. "['L194', 'L195']"
                line_ids_str = parts[3].strip()
                line_ids = [lid.strip("'\" ") for lid in line_ids_str[1:-1].split(",")]
                
                # Pair consecutive lines as Q&A
                for i in range(len(line_ids) - 1):
                    q_id = line_ids[i]
                    a_id = line_ids[i+1]
                    
                    if q_id in lines and a_id in lines:
                        q_text = lines[q_id]
                        a_text = lines[a_id]
                        
                        # Skip empty lines
                        if q_text and a_text:
                            out.write(f"Question: {q_text}\nAnswer: {a_text}\n***\n")
                            qa_count += 1
                            
    print(f"Compilation complete! Generated {qa_count} QA pairs saved to {output_file}.")
    
    # Clean up raw extracted files to keep workspace tidy
    print("Cleaning up extracted directories...")
    import shutil
    # find the extracted directory to delete
    extracted_root = os.path.dirname(lines_file)
    shutil.rmtree(extracted_root)
    print("Cleanup complete. Ready for training!")

if __name__ == "__main__":
    download_and_extract()
