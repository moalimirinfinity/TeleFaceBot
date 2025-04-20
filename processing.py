import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import numpy as np
from numpy.linalg import norm
import heapq
import logging # Import logging

logger = logging.getLogger(__name__) # Setup logger for this module

def initialize_models():
    """Initializes and returns the MTCNN and InceptionResnetV1 models."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Use logger instead of print
    logger.info(f"Initializing models on device: {device}")
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, select_largest=False, keep_all=True,
        device=device
    )
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return mtcnn, resnet, device

def get_embeddings(image_path, mtcnn, resnet, device):
    """Extracts face embeddings from an image file."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            # Use logger
            logger.warning(f"Could not read image file: {image_path}")
            return np.array([])
        # MTCNN expects RGB, cv2 reads BGR
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x_aligned = mtcnn(image_rgb)
        if x_aligned is not None:
            # Ensure x_aligned is a tensor; handle cases where multiple faces might be detected
            if isinstance(x_aligned, list):
                 x_aligned = torch.stack(x_aligned)
            
            embeddings = resnet(x_aligned.to(device)).detach().cpu().numpy()
            logger.debug(f"Found {len(embeddings)} embeddings in {image_path}")
            return embeddings # Return all detected embeddings
        logger.debug(f"No faces detected by MTCNN in {image_path}")
        return np.array([])
    except Exception as e:
        # Use logger, include stack trace with exc_info=True
        logger.error(f"Error processing image {image_path}: {e}", exc_info=True)
        return np.array([])

def cosine_similarity(A, B):
    """Calculates the cosine similarity between two vectors or matrices."""
    # Ensure inputs are numpy arrays
    A = np.asarray(A)
    B = np.asarray(B)
    
    # Handle single vector vs single vector
    if A.ndim == 1 and B.ndim == 1:
        return np.dot(A, B) / (norm(A) * norm(B))
    
    # Handle matrix vs matrix (row-wise similarity)
    elif A.ndim == 2 and B.ndim == 2:
        # Ensure A and B have the same number of columns (embedding dimension)
        if A.shape[1] != B.shape[1]:
             raise ValueError("Matrices A and B must have the same number of columns.")
        # Calculate dot products (row-wise)
        dot_products = np.sum(A * B, axis=1)
        # Calculate norms (row-wise)
        norm_A = norm(A, axis=1)
        norm_B = norm(B, axis=1)
        # Avoid division by zero
        zero_norm_mask = (norm_A == 0) | (norm_B == 0)
        similarities = np.zeros_like(dot_products, dtype=float)
        if not np.all(zero_norm_mask):
             similarities[~zero_norm_mask] = dot_products[~zero_norm_mask] / (norm_A[~zero_norm_mask] * norm_B[~zero_norm_mask])
        return similarities

    # Handle matrix vs vector (compare vector B against each row in matrix A)
    elif A.ndim == 2 and B.ndim == 1:
        if A.shape[1] != B.shape[0]:
            raise ValueError("Matrix A columns must match vector B length.")
        dot_products = A @ B
        norm_A = norm(A, axis=1)
        norm_B = norm(B)
        zero_norm_mask = (norm_A == 0) | (norm_B == 0)
        similarities = np.zeros_like(dot_products, dtype=float)
        if not np.all(zero_norm_mask):
             similarities[~zero_norm_mask] = dot_products[~zero_norm_mask] / (norm_A[~zero_norm_mask] * norm_B)
        return similarities
        
    else:
        raise ValueError("Unsupported input dimensions for cosine_similarity")

def find_similars(query_embeddings, index_data):
    """Finds images in the index similar to the query embeddings."""
    if not index_data:
        # Use logger
        logger.warning("Index data is empty. Cannot find similar images.")
        return []
        
    similars = []
    # index_data is expected to be {embedding_tuple: (chat_id, message_id)}
    index_embeddings = list(index_data.keys()) # List of embedding tuples
    index_np = np.array([list(emb) for emb in index_embeddings]) # Convert to NumPy array

    if not index_np.size > 0:
        # Use logger
        logger.warning("Index embeddings array is empty after loading keys.")
        return []
        
    # Calculate similarity between each query embedding and all index embeddings
    # We want the *maximum* similarity for any query embedding against each indexed image
    max_similarities = np.zeros(len(index_np))
    
    for query_emb in query_embeddings:
        similarities = cosine_similarity(index_np, query_emb) # Compare one query emb against all index embs
        max_similarities = np.maximum(max_similarities, similarities) # Keep the highest similarity found so far for each index entry

    # Use max_similarities for ranking
    for i, score in enumerate(max_similarities):
        # Use negative score because heapq is a min-heap
        similars.append((-score, index_embeddings[i]))

    heapq.heapify(similars)
    
    # Yield unique images based on similarity score
    image_set = set()
    results = []
    while similars:
        score, emb_tuple = heapq.heappop(similars)
        image_location = index_data[emb_tuple]
        if image_location in image_set:
            continue
        image_set.add(image_location)
        # Yield score along with location if needed, or just location
        results.append({'score': -score, 'location': image_location}) # Store score and location

    # Sort results by score descending (optional, heapq already gives some order)
    results.sort(key=lambda x: x['score'], reverse=True)
    
    logger.info(f"Found {len(results)} unique similar images.")
    # Return only the locations in order
    return [item['location'] for item in results] 