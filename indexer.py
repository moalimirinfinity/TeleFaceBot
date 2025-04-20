import os
import pickle
import logging
import json
from dotenv import load_dotenv
from tqdm import tqdm

# Import necessary functions from processing.py
from processing import initialize_models, get_embeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- Configuration --- 
# Input directory for images
DOWNLOADER_OUTPUT_DIR = os.getenv("DOWNLOADER_OUTPUT_DIR", "data")
INDEXER_INPUT_DIR = os.getenv("INDEXER_INPUT_DIR", DOWNLOADER_OUTPUT_DIR)

# Target Chat ID (used for storing message location)
TARGET_CHAT_ID_STR = os.getenv("DOWNLOADER_TARGET_CHAT_ID")

# Output index file path
INDEX_FILES_STR = os.getenv("INDEX_FILES", "")
INDEXER_OUTPUT_FILE_OVERRIDE = os.getenv("INDEXER_OUTPUT_FILE")

# --- Validation --- 
if not INDEXER_INPUT_DIR or not os.path.isdir(INDEXER_INPUT_DIR):
    logger.error(f"Input directory not found or not specified: {INDEXER_INPUT_DIR}")
    logger.error("Ensure DOWNLOADER_OUTPUT_DIR or INDEXER_INPUT_DIR is set in .env and points to a valid directory.")
    exit(1)

if not TARGET_CHAT_ID_STR:
     logger.error("DOWNLOADER_TARGET_CHAT_ID must be set in the .env file (used to store message origin).")
     exit(1)

# Determine the output file path
output_file_path = None
if INDEXER_OUTPUT_FILE_OVERRIDE:
    output_file_path = INDEXER_OUTPUT_FILE_OVERRIDE
    logger.info(f"Using output file path override: {output_file_path}")
elif INDEX_FILES_STR:
    try:
        first_index_file = INDEX_FILES_STR.split(',')[0].strip()
        if first_index_file:
            output_file_path = first_index_file
            logger.info(f"Using first path from INDEX_FILES as output: {output_file_path}")
        else:
             raise ValueError("First path in INDEX_FILES is empty.")
    except Exception as e:
         logger.error(f"Could not determine output file path from INDEX_FILES='{INDEX_FILES_STR}': {e}")
         exit(1)
else:
    logger.error("No output file path specified. Set INDEXER_OUTPUT_FILE or INDEX_FILES in .env.")
    exit(1)
    
# Try converting TARGET_CHAT_ID to int if possible, otherwise keep as string (for usernames)
try:
    TARGET_CHAT_ID = int(TARGET_CHAT_ID_STR)
except ValueError:
    TARGET_CHAT_ID = TARGET_CHAT_ID_STR
    logger.info(f"Target chat ID '{TARGET_CHAT_ID}' is not an integer, using as string.")

# --- Main Logic --- 
def run_indexer():
    logger.info("Initializing models for indexing...")
    try:
        mtcnn, resnet, device = initialize_models()
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}", exc_info=True)
        return

    embedding_to_location = {}
    processed_files = 0
    skipped_files = 0
    error_files = 0

    logger.info(f"Starting indexing process for directory: {INDEXER_INPUT_DIR}")
    logger.info(f"Output will be saved to: {output_file_path}")
    logger.info(f"Messages will be associated with Chat ID: {TARGET_CHAT_ID}")

    try:
        # List files, filter for common image types (optional but good practice)
        image_files = [f for f in os.listdir(INDEXER_INPUT_DIR) 
                       if os.path.isfile(os.path.join(INDEXER_INPUT_DIR, f)) and 
                       f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        logger.info(f"Found {len(image_files)} potential image files in {INDEXER_INPUT_DIR}")
        
        for filename in tqdm(sorted(image_files), desc="Indexing Images"):
            file_path = os.path.join(INDEXER_INPUT_DIR, filename)
            try:
                # Extract message ID from filename (assuming filename is message_id.jpg)
                message_id_str = os.path.splitext(filename)[0]
                message_id = int(message_id_str)
                
                # Get embeddings for the image
                # Note: get_embeddings returns an array of embeddings (one per face)
                embeddings = get_embeddings(file_path, mtcnn, resnet, device)
                
                if embeddings.size == 0:
                    logger.debug(f"No faces found in {filename}, skipping.")
                    skipped_files += 1
                    continue

                # Store each embedding with its location
                num_faces = 0
                for embedding in embeddings:
                    # Convert numpy array to tuple for dictionary key
                    embedding_tuple = tuple(embedding)
                    location = (TARGET_CHAT_ID, message_id)
                    
                    # Handle potential hash collisions (rare but possible)
                    if embedding_tuple in embedding_to_location and embedding_to_location[embedding_tuple] != location:
                         logger.warning(f"Embedding collision detected for message {message_id}. Overwriting previous entry {embedding_to_location[embedding_tuple]} for the same embedding.")
                         
                    embedding_to_location[embedding_tuple] = location
                    num_faces += 1
                
                logger.debug(f"Processed {filename}: Found {num_faces} face(s).")
                processed_files += 1

            except ValueError:
                 logger.warning(f"Could not parse message ID from filename: {filename}. Skipping.")
                 skipped_files += 1
            except Exception as e:
                logger.error(f"Error processing file {filename}: {e}", exc_info=True)
                error_files += 1

    except Exception as e:
         logger.error(f"An error occurred during directory traversal or processing: {e}", exc_info=True)
         return # Abort saving if a major error occurred

    logger.info(f"Indexing finished. Processed: {processed_files}, Skipped: {skipped_files}, Errors: {error_files}")
    logger.info(f"Total unique embeddings found: {len(embedding_to_location)}")

    # Save the index
    if not embedding_to_location:
        logger.warning("No embeddings were generated. Index file will not be saved.")
        return
        
    logger.info(f"Saving index to {output_file_path}...")
    try:
        # Ensure parent directory exists for the output file
        output_dir = os.path.dirname(output_file_path)
        if output_dir:
             os.makedirs(output_dir, exist_ok=True)
             
        with open(output_file_path, 'wb') as f:
            pickle.dump(embedding_to_location, f)
        logger.info("Index saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save index file {output_file_path}: {e}", exc_info=True)

if __name__ == "__main__":
    run_indexer() 