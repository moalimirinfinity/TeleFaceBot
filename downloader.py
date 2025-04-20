import os
import json
import logging
from pyrogram import Client
from pyrogram.errors import FloodWait, ChannelInvalid, ChannelPrivate, ChatAdminRequired, UserNotParticipant
from dotenv import load_dotenv
import asyncio

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

API_ID = os.getenv("API_ID")
API_HASH = os.getenv("API_HASH")
PROXY_STR = os.getenv("PROXY")
TARGET_CHAT_ID_STR = os.getenv("DOWNLOADER_TARGET_CHAT_ID")
OUTPUT_DIR = os.getenv("DOWNLOADER_OUTPUT_DIR", "data") # Default to "data"

# Parse PROXY string
proxy = None
if PROXY_STR:
    try:
        proxy = json.loads(PROXY_STR)
    except json.JSONDecodeError:
        logger.warning(f"Could not parse PROXY environment variable: {PROXY_STR}")
        proxy = None

# Validate required variables
if not all([API_ID, API_HASH, TARGET_CHAT_ID_STR]):
    logger.error("API_ID, API_HASH, and DOWNLOADER_TARGET_CHAT_ID must be set in the .env file.")
    exit(1)

# Try converting API_ID to int
try:
    API_ID = int(API_ID)
except ValueError:
    logger.error("API_ID must be an integer.")
    exit(1)

# Try converting TARGET_CHAT_ID to int if possible, otherwise keep as string (for usernames)
try:
    TARGET_CHAT_ID = int(TARGET_CHAT_ID_STR)
except ValueError:
    TARGET_CHAT_ID = TARGET_CHAT_ID_STR
    logger.info(f"Target chat ID '{TARGET_CHAT_ID}' is not an integer, treating as username/invite link.")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
logger.info(f"Ensured output directory exists: {OUTPUT_DIR}")

# Use a user session name (will create my_account.session)
app = Client("my_account", api_id=API_ID, api_hash=API_HASH, proxy=proxy)

async def download_photos():
    download_count = 0
    skipped_count = 0 # Already downloaded
    skipped_type_count = 0 # Non-photo messages
    error_count = 0
    total_messages = 0

    async with app:
        logger.info(f"Starting download from chat: {TARGET_CHAT_ID}")
        try:
            # Check if chat exists and we can access it
            chat = await app.get_chat(TARGET_CHAT_ID)
            logger.info(f"Successfully accessed chat: {chat.title or chat.first_name} ({chat.id})")
            
            async for message in app.get_chat_history(TARGET_CHAT_ID):
                total_messages += 1
                if total_messages % 100 == 0:
                    logger.info(f"Processed {total_messages} messages...")
                    
                if message.photo:
                    # Construct filename: output_dir/message_id.jpg
                    file_path = os.path.join(OUTPUT_DIR, f"{message.id}.jpg")
                    
                    # Check if file already exists to avoid re-downloading
                    if os.path.exists(file_path):
                        # logger.debug(f"Skipping already downloaded photo: {file_path}")
                        skipped_count += 1
                        continue
                        
                    try:
                        await app.download_media(message.photo.file_id, file_name=file_path)
                        logger.info(f"Downloaded photo: {file_path} (Message ID: {message.id})")
                        download_count += 1
                    except FloodWait as e:
                        logger.warning(f"Rate limited. Waiting for {e.value} seconds...")
                        await asyncio.sleep(e.value + 1) # Wait a bit longer
                        # Retry downloading the same message
                        try:
                             await app.download_media(message.photo.file_id, file_name=file_path)
                             logger.info(f"Downloaded photo after wait: {file_path}")
                             download_count += 1
                        except Exception as retry_e:
                             logger.error(f"Failed to download photo {message.id} after FloodWait: {retry_e}")
                             error_count += 1
                    except Exception as e:
                        logger.error(f"Failed to download photo from message {message.id}: {e}")
                        error_count += 1
                else:
                    logger.debug(f"Skipping non-photo message {message.id}")
                    skipped_type_count += 1

        except (ChannelInvalid, ChannelPrivate, ChatAdminRequired, UserNotParticipant) as e:
             logger.error(f"Cannot access chat {TARGET_CHAT_ID}: {e}. Please check the chat ID and your permissions.")
             error_count += 1
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            error_count += 1
        finally:
            logger.info("Download process finished.")
            logger.info(f"Summary: Processed={total_messages}, Downloaded={download_count}, Skipped (exists)={skipped_count}, Skipped (type)={skipped_type_count}, Errors={error_count}")

if __name__ == "__main__":
    logger.info("Running downloader script...")
    app.run(download_photos())