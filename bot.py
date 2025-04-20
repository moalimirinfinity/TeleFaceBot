from pyrogram import Client, filters
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton
import pickle
import os
import json
import numpy as np # Keep numpy import here for random
from dotenv import load_dotenv
import logging # Import logging
import asyncio # Import asyncio

# Import our processing functions
from processing import initialize_models, get_embeddings, find_similars

load_dotenv()  # Load variables from .env file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler() # Log to console
        # You could add logging.FileHandler("bot.log") here to log to a file
    ]
)
logger = logging.getLogger(__name__)

# Initialize models from processing.py
mtcnn, resnet, device = initialize_models()

# Load configuration from environment variables
API_ID = os.getenv("API_ID")
API_HASH = os.getenv("API_HASH")
BOT_TOKEN = os.getenv("BOT_TOKEN")
INDEX_FILES_STR = os.getenv("INDEX_FILES", "")  # Default to empty string if not set
PROXY_STR = os.getenv("PROXY")

# Parse PROXY string if it exists
proxy = None
if PROXY_STR:
    try:
        proxy = json.loads(PROXY_STR) # Expecting a JSON dictionary string
    except json.JSONDecodeError:
        print(f"Warning: Could not parse PROXY environment variable: {PROXY_STR}")
        proxy = None

# Check if required variables are set
if not all([API_ID, API_HASH, BOT_TOKEN]):
    raise ValueError("API_ID, API_HASH, and BOT_TOKEN must be set in the .env file.")

# Try converting API_ID to int
try:
    API_ID = int(API_ID)
except (ValueError, TypeError):
     raise ValueError("API_ID must be an integer.")

app = Client("galley_bot", api_id=API_ID, api_hash=API_HASH,
             bot_token=BOT_TOKEN, proxy=proxy)

message_embeddings = {}
# Add a dictionary to store locks for media groups
media_group_locks = {}

# Load index files from configuration
indexes = [index.strip() for index in INDEX_FILES_STR.split(',') if index.strip()]
embedding_to_file = {}
for index_path in indexes:
    try:
        with open(index_path, 'rb') as f:
            embedding_to_file_partial = pickle.load(f)
            embedding_to_file.update(embedding_to_file_partial)
            logger.info(f"Successfully loaded index: {index_path}")
    except FileNotFoundError:
        logger.warning(f"Index file not found: {index_path}")
    except Exception as e:
        logger.warning(f"Failed to load index file {index_path}: {e}")

if not embedding_to_file:
    logger.warning("No index data loaded. The bot might not find similar images.")

# Load Text Configuration
TEXT_DEFAULTS = {
    "TEXT_START_HELP": "Send me a photo of yourself, and I'll find similar photos from the event.",
    "TEXT_PHOTO_RECEIVED": "Photo received. Click the button below to view similar images.",
    "TEXT_PROCESSING_ERROR": "An error occurred while processing the image.",
    "TEXT_NO_FACE_FOUND": "Unfortunately, no face could be detected in the submitted photo.",
    "TEXT_LINK_EXPIRED": "This link has expired. Please send the photo again.",
    "TEXT_NO_MORE_SIMILAR": "No more similar images found.",
    "TEXT_VIEW_MORE_BUTTON": "View More Images {current_count}/{next_count}",
    "TEXT_VIEW_SIMILAR_BUTTON": "View Similar Images 0/10",
    "TEXT_SHOWING_SIMILAR": "Showing {count} similar images.",
    "TEXT_CLICK_BELOW_MORE": "Click the button below to see more images."
}
TEXT_CONFIG = {key: os.getenv(key, default_val) for key, default_val in TEXT_DEFAULTS.items()}

@app.on_callback_query()
async def continue_find(client, callback_query):
    # --- Try block covering the entire callback logic --- 
    try:
        # --- Code inside the try block is indented --- 
        key, end = callback_query.data.split('|')
        key, end = int(key), int(end)
        start = end - 10
        logger.info(f"Callback query received for key {key}, range {start}-{end}")
        
        if key not in message_embeddings:
            await callback_query.answer(TEXT_CONFIG["TEXT_LINK_EXPIRED"])
            logger.warning(f"Expired callback key received: {key}")
            return
        
        embeddings = message_embeddings[key]
        await callback_query.answer() # Acknowledge the callback
        
        # Pass index data (embedding_to_file) to find_similars
        similar_images_list = find_similars(embeddings, embedding_to_file)
        
        paginated_results = similar_images_list[start:end]

        if not paginated_results:
            await callback_query.message.edit_text(text=TEXT_CONFIG["TEXT_NO_MORE_SIMILAR"])
            logger.info(f"No more similar images found for key {key} beyond index {start}")
            return
            
        logger.info(f"Forwarding {len(paginated_results)} images for key {key}, range {start}-{end}")
        for similar_image_location in paginated_results:
            chat_id, message_id = similar_image_location
            try:
                await client.forward_messages(callback_query.message.chat.id, chat_id, message_id)
            except Exception as e:
                 logger.error(f"Failed to forward message {message_id} from chat {chat_id}: {e}")
                 # Optionally notify user about forwarding error
                 # await client.send_message(callback_query.message.chat.id, f"Could not retrieve one of the images.")

        # Update button based on whether more results exist
        if end < len(similar_images_list):
            button_text = TEXT_CONFIG["TEXT_VIEW_MORE_BUTTON"].format(current_count=end, next_count=end+10)
            await callback_query.message.edit_text(text=TEXT_CONFIG["TEXT_CLICK_BELOW_MORE"],
                                           reply_markup=InlineKeyboardMarkup([[
                                                    InlineKeyboardButton(button_text,
                                                                         callback_data=f'{key}|{end+10}')
                                           ]]))
        else:
            final_text = TEXT_CONFIG["TEXT_SHOWING_SIMILAR"].format(count=len(similar_images_list))
            await callback_query.message.edit_text(text=final_text)
            logger.info(f"Finished showing all {len(similar_images_list)} images for key {key}")
            # Optional cleanup
            # if key in message_embeddings:
            #     del message_embeddings[key]
            #     logger.info(f"Cleaned up embeddings for key {key}")
    
    # --- Except block, correctly indented, catching errors from the try block ---          
    except Exception as e:
        logger.error(f"Error processing callback query {callback_query.id}: {e}", exc_info=True)
        try:
             await callback_query.answer("An internal error occurred.") # Generic error to user
        except Exception as inner_e:
             logger.error(f"Failed to answer callback query {callback_query.id} after error: {inner_e}")

@app.on_message(filters.private & filters.photo)
async def start_find(client, message):
    download_path = None
    user_id = message.from_user.id
    message_id = message.id
    media_group_id = message.media_group_id
    log_prefix = f"User {user_id}, Msg {message_id}:"
    logger.info(f"{log_prefix} Received photo (Media Group: {media_group_id})")
    
    lock = None # Initialize lock variable
    
    try:
        photo = message.photo
        download_path = await client.download_media(photo.file_id, file_name=f"{photo.file_id}.jpg") 
        logger.info(f"{log_prefix} Downloaded photo to {download_path}")
        
        embeddings = get_embeddings(download_path, mtcnn, resnet, device)
        
        if embeddings.size == 0:
            logger.warning(f"{log_prefix} No face detected in {download_path}")
            await client.send_message(chat_id=message.chat.id, text=TEXT_CONFIG["TEXT_NO_FACE_FOUND"])
            return
            
        logger.info(f"{log_prefix} Detected {len(embeddings)} face(s) in {download_path}")
        detected_embeddings = [emb for emb in embeddings]

        key = int(np.random.randint(1_000_000_000)) if media_group_id is None else int(media_group_id)

        # --- Lock acquisition for media groups --- 
        send_button = False
        if media_group_id:
            # Get or create lock for this media group
            if key not in media_group_locks:
                 media_group_locks[key] = asyncio.Lock()
            lock = media_group_locks[key]
            
            async with lock:
                if key not in message_embeddings:
                     message_embeddings[key] = []
                     send_button = True # First message for this group to acquire lock
                message_embeddings[key].extend(detected_embeddings)
                logger.info(f"{log_prefix} Added {len(detected_embeddings)} embeddings to locked key {key}. Total: {len(message_embeddings[key])}")
        else: # Single message, no lock needed
             message_embeddings[key] = detected_embeddings # Store directly
             send_button = True # Always send button for single messages
             logger.info(f"{log_prefix} Stored {len(detected_embeddings)} new embeddings for single message key {key}")
        # --- End Lock Handling --- 

        if send_button:
            await client.send_message(chat_id=message.chat.id, 
                                    text=TEXT_CONFIG["TEXT_PHOTO_RECEIVED"],
                                      reply_markup=InlineKeyboardMarkup([[
                                          InlineKeyboardButton(
                                            TEXT_CONFIG["TEXT_VIEW_SIMILAR_BUTTON"],
                                            callback_data=f'{key}|10')
                                    ]]))
            logger.info(f"{log_prefix} Sent confirmation and button for key {key}")
            
    except Exception as e:
        logger.error(f"{log_prefix} Error in start_find: {e}", exc_info=True)
        await client.send_message(chat_id=message.chat.id, text=TEXT_CONFIG["TEXT_PROCESSING_ERROR"])
    finally:
        if download_path and os.path.exists(download_path):
            try:
                os.remove(download_path)
                logger.info(f"{log_prefix} Removed temporary file: {download_path}")
            except OSError as e:
                 logger.error(f"{log_prefix} Error removing temporary file {download_path}: {e}")
        
        # Clean up lock if it exists and the key is likely done (might need better cleanup strategy)
        # This simple cleanup assumes the group is processed; consider TTL if needed
        # if lock and key in media_group_locks and not lock.locked(): 
        #    # Check if embeddings exist before deleting lock? Potential race condition.
        #    # For simplicity, let's not auto-delete locks here. They'll persist.
        #    pass 

@app.on_message(filters.private & filters.command(["start", "help"]))
async def my_handler(client, message):
    user_id = message.from_user.id
    logger.info(f"Received /start or /help command from user {user_id}")
    await client.send_message(chat_id=message.chat.id, text=TEXT_CONFIG["TEXT_START_HELP"])

logger.info("Starting bot...")
app.run()
logger.info("Bot stopped.")
