"""
Flask routes for Favorites API

Provides REST endpoints for managing user favorites (bookmarked media outputs).
Favorites persist across sessions and allow restoring complete session settings.

Session 127: Initial Implementation
Based on: docs/PARKED_reflexion_feature_konzept.md (Persistent Footer Gallery)
"""
import json
import logging
from datetime import datetime
from flask import Blueprint, jsonify, request
from pathlib import Path

from config import JSON_STORAGE_DIR
from my_app.services.pipeline_recorder import load_recorder

logger = logging.getLogger(__name__)

# Blueprint erstellen
favorites_bp = Blueprint('favorites', __name__, url_prefix='/api/favorites')

# Favorites storage file
FAVORITES_FILE = JSON_STORAGE_DIR / "favorites.json"


def _load_favorites() -> dict:
    """
    Load favorites from disk.

    Returns:
        dict with version, mode, and favorites list
    """
    if not FAVORITES_FILE.exists():
        return {
            "version": "1.0",
            "mode": "global",
            "favorites": []
        }

    try:
        with open(FAVORITES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"[FAVORITES] Error loading favorites: {e}")
        return {
            "version": "1.0",
            "mode": "global",
            "favorites": []
        }


def _save_favorites(data: dict) -> bool:
    """
    Save favorites to disk.

    Args:
        data: Favorites data dict

    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        FAVORITES_FILE.parent.mkdir(parents=True, exist_ok=True)

        with open(FAVORITES_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"[FAVORITES] Error saving favorites: {e}")
        return False


def _get_thumbnail_url(run_id: str, media_type: str) -> str:
    """
    Generate thumbnail URL for a given run and media type.

    Args:
        run_id: Run ID
        media_type: Media type (image, audio, video, etc.)

    Returns:
        URL string for the thumbnail
    """
    if media_type == 'image':
        return f"/api/media/image/{run_id}/0"
    elif media_type == 'video':
        return f"/api/media/video/{run_id}"
    elif media_type in ['audio', 'music']:
        return f"/api/media/audio/{run_id}"
    else:
        return f"/api/media/{media_type}/{run_id}"


@favorites_bp.route('', methods=['GET'])
def get_favorites():
    """
    List all favorites with metadata.

    Query parameters:
        - device_id: Filter favorites by device ID (optional)
        - view_mode: 'per_user' (filter by device_id) or 'global' (show all)

    Returns:
        200: List of favorites with thumbnail URLs and metadata
        500: Server error

    Example:
        GET /api/favorites?device_id=abc123_2026-01-28&view_mode=per_user
    """
    try:
        # Query parameters for filtering
        device_id = request.args.get('device_id')
        view_mode = request.args.get('view_mode', 'per_user')  # 'per_user' or 'global'

        logger.info(f"[FAVORITES] Fetching favorites (view_mode={view_mode}, device_id={device_id})")

        data = _load_favorites()
        favorites = data.get('favorites', [])

        # Filter by device_id if in per_user mode
        if view_mode == 'per_user' and device_id:
            favorites = [f for f in favorites if f.get('device_id') == device_id]
            logger.info(f"[FAVORITES] Filtered to {len(favorites)} favorites for device {device_id}")
        else:
            logger.info(f"[FAVORITES] Returning {len(favorites)} favorites (global mode)")

        # Enrich favorites with additional metadata from run data
        enriched_favorites = []
        for fav in favorites:
            run_id = fav.get('run_id')

            # Try to load run metadata
            recorder = load_recorder(run_id, base_path=JSON_STORAGE_DIR)

            enriched = {
                **fav,
                'exists': recorder is not None
            }

            if recorder:
                # Add run metadata
                enriched['schema'] = recorder.metadata.get('config_name', 'unknown')
                enriched['timestamp'] = recorder.metadata.get('timestamp', '')

                # Get input text if available
                for entity in recorder.metadata.get('entities', []):
                    if entity['type'] == 'input':
                        input_file = recorder.final_folder / entity['filename']
                        if input_file.exists():
                            text = input_file.read_text(encoding='utf-8')
                            # Truncate for preview
                            enriched['input_preview'] = text[:100] + '...' if len(text) > 100 else text
                            break

            enriched_favorites.append(enriched)

        logger.info(f"[FAVORITES] Returning {len(enriched_favorites)} favorites")

        return jsonify({
            'favorites': enriched_favorites,
            'total': len(enriched_favorites),
            'mode': data.get('mode', 'global')
        }), 200

    except Exception as e:
        logger.error(f"[FAVORITES] Error fetching favorites: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'Failed to fetch favorites',
            'details': str(e)
        }), 500


@favorites_bp.route('', methods=['POST'])
def add_favorite():
    """
    Add a new favorite.

    Request body (JSON):
        - run_id: Run ID to favorite (required)
        - media_type: Type of media ('image', 'audio', 'video', etc.) (required)
        - device_id: Device identifier (browser_id + date) (optional)
        - user_id: Optional user ID (default: 'anonymous')
        - user_note: Optional note from user

    Returns:
        201: Favorite created successfully
        400: Invalid request (missing required fields)
        409: Favorite already exists
        500: Server error

    Example:
        POST /api/favorites
        {"run_id": "run_abc123", "media_type": "image"}
    """
    try:
        # Parse request
        body = request.get_json()

        if not body:
            return jsonify({'error': 'Request body required'}), 400

        run_id = body.get('run_id')
        media_type = body.get('media_type')
        device_id = body.get('device_id')  # Session 145: Per-user favorites
        user_id = body.get('user_id', 'anonymous')
        user_note = body.get('user_note', '')
        source_view = body.get('source_view')  # Vue route path for correct restore routing

        # Session 145: Debug logging
        logger.info(f"[FAVORITES] POST Request Body: {body}")
        logger.info(f"[FAVORITES] device_id extracted: {device_id!r} (type: {type(device_id).__name__})")

        if not run_id:
            return jsonify({'error': 'run_id is required'}), 400

        if not media_type:
            return jsonify({'error': 'media_type is required'}), 400

        logger.info(f"[FAVORITES] Adding favorite: run_id={run_id}, media_type={media_type}, device_id={device_id}")

        # Verify run exists
        recorder = load_recorder(run_id, base_path=JSON_STORAGE_DIR)
        if not recorder:
            return jsonify({
                'error': 'Run not found',
                'run_id': run_id
            }), 404

        # Load existing favorites
        data = _load_favorites()
        favorites = data.get('favorites', [])

        # Check if already exists
        for fav in favorites:
            if fav.get('run_id') == run_id:
                return jsonify({
                    'error': 'Favorite already exists',
                    'run_id': run_id
                }), 409

        # Create new favorite entry
        new_favorite = {
            'run_id': run_id,
            'device_id': device_id,  # Session 145: Per-user favorites
            'added_at': datetime.now().isoformat(),
            'thumbnail_url': _get_thumbnail_url(run_id, media_type),
            'media_type': media_type,
            'user_id': user_id,
            'user_note': user_note,
            'source_view': source_view
        }

        # Add to beginning of list (most recent first)
        favorites.insert(0, new_favorite)
        data['favorites'] = favorites

        # Save
        if not _save_favorites(data):
            return jsonify({'error': 'Failed to save favorite'}), 500

        logger.info(f"[FAVORITES] Successfully added favorite: {run_id}")

        return jsonify({
            'success': True,
            'favorite': new_favorite,
            'total': len(favorites)
        }), 201

    except Exception as e:
        logger.error(f"[FAVORITES] Error adding favorite: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'Failed to add favorite',
            'details': str(e)
        }), 500


@favorites_bp.route('/<run_id>', methods=['DELETE'])
def remove_favorite(run_id: str):
    """
    Remove a favorite by run_id.

    Args:
        run_id: Run ID to remove from favorites

    Returns:
        200: Favorite removed successfully
        404: Favorite not found
        500: Server error

    Example:
        DELETE /api/favorites/run_abc123
    """
    try:
        logger.info(f"[FAVORITES] Removing favorite: {run_id}")

        # Load existing favorites
        data = _load_favorites()
        favorites = data.get('favorites', [])

        # Find and remove
        original_count = len(favorites)
        favorites = [f for f in favorites if f.get('run_id') != run_id]

        if len(favorites) == original_count:
            return jsonify({
                'error': 'Favorite not found',
                'run_id': run_id
            }), 404

        data['favorites'] = favorites

        # Save
        if not _save_favorites(data):
            return jsonify({'error': 'Failed to save favorites'}), 500

        logger.info(f"[FAVORITES] Successfully removed favorite: {run_id}")

        return jsonify({
            'success': True,
            'run_id': run_id,
            'total': len(favorites)
        }), 200

    except Exception as e:
        logger.error(f"[FAVORITES] Error removing favorite: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'Failed to remove favorite',
            'details': str(e)
        }), 500


@favorites_bp.route('/<run_id>/restore', methods=['GET'])
def get_restore_data(run_id: str):
    """
    Get complete session data for restoring a favorite.

    Returns all data needed to restore the session state:
    - Original input text
    - Schema/config used
    - Settings (safety level, etc.)
    - Media URLs

    Args:
        run_id: Run ID to get restore data for

    Returns:
        200: Complete restore data
        404: Run not found
        500: Server error

    Example:
        GET /api/favorites/run_abc123/restore
    """
    try:
        logger.info(f"[FAVORITES] Getting restore data for: {run_id}")

        # Load run data
        recorder = load_recorder(run_id, base_path=JSON_STORAGE_DIR)

        if not recorder:
            return jsonify({
                'error': 'Run not found',
                'run_id': run_id
            }), 404

        metadata = recorder.metadata
        entities = metadata.get('entities', [])

        # Build restore data
        restore_data = {
            'run_id': run_id,
            'schema': metadata.get('config_name', 'unknown'),
            'timestamp': metadata.get('timestamp', ''),
            'current_state': metadata.get('current_state', {}),
            'expected_outputs': metadata.get('expected_outputs', []),
            'user_id': metadata.get('user_id', 'anonymous')
        }

        # Get input text
        for entity in entities:
            if entity['type'] == 'input':
                input_file = recorder.final_folder / entity['filename']
                if input_file.exists():
                    restore_data['input_text'] = input_file.read_text(encoding='utf-8')
                break

        # Get context prompt (meta-prompt/rules - user-editable!)
        for entity in entities:
            if entity['type'] == 'context_prompt':
                ctx_file = recorder.final_folder / entity['filename']
                if ctx_file.exists():
                    restore_data['context_prompt'] = ctx_file.read_text(encoding='utf-8')
                break

        # Get transformed/intercepted text if available
        for entity in entities:
            if entity['type'] == 'interception':
                output_file = recorder.final_folder / entity['filename']
                if output_file.exists():
                    restore_data['transformed_text'] = output_file.read_text(encoding='utf-8')
                break

        # Get English translation if available
        for entity in entities:
            if entity['type'] == 'translation_en':
                trans_file = recorder.final_folder / entity['filename']
                if trans_file.exists():
                    restore_data['translation_en'] = trans_file.read_text(encoding='utf-8')
                break

        # Include models used at each stage
        if 'models_used' in metadata:
            restore_data['models_used'] = metadata['models_used']

        # Get media URLs
        media_outputs = []
        for entity in entities:
            entity_type = entity.get('type', '')
            if entity_type.startswith('output_'):
                media_type = entity_type.replace('output_', '')
                media_outputs.append({
                    'type': media_type,
                    'filename': entity.get('filename'),
                    'url': _get_thumbnail_url(run_id, media_type),
                    'metadata': entity.get('metadata', {})
                })

        restore_data['media_outputs'] = media_outputs

        # Determine target view: use stored source_view if available, else heuristic
        data = _load_favorites()
        fav_entry = next((f for f in data.get('favorites', []) if f.get('run_id') == run_id), None)

        if fav_entry and fav_entry.get('source_view'):
            restore_data['target_view'] = fav_entry['source_view']
        else:
            # Fallback heuristic for old favorites without source_view
            has_input_image = any(
                entity.get('type') == 'input_image'
                for entity in entities
            )
            if has_input_image:
                restore_data['target_view'] = 'image-transformation'
            else:
                restore_data['target_view'] = 'text-transformation'

        logger.info(f"[FAVORITES] Successfully retrieved restore data for: {run_id}")

        return jsonify(restore_data), 200

    except Exception as e:
        logger.error(f"[FAVORITES] Error getting restore data: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'Failed to get restore data',
            'details': str(e)
        }), 500


@favorites_bp.route('/<run_id>', methods=['PATCH'])
def update_favorite(run_id: str):
    """
    Update a favorite's metadata (e.g., user_note).

    Request body (JSON):
        - user_note: New note for the favorite (optional)

    Returns:
        200: Favorite updated successfully
        404: Favorite not found
        500: Server error

    Example:
        PATCH /api/favorites/run_abc123
        {"user_note": "My best creation!"}
    """
    try:
        body = request.get_json() or {}

        logger.info(f"[FAVORITES] Updating favorite: {run_id}")

        # Load existing favorites
        data = _load_favorites()
        favorites = data.get('favorites', [])

        # Find and update
        found = False
        for fav in favorites:
            if fav.get('run_id') == run_id:
                if 'user_note' in body:
                    fav['user_note'] = body['user_note']
                found = True
                break

        if not found:
            return jsonify({
                'error': 'Favorite not found',
                'run_id': run_id
            }), 404

        # Save
        if not _save_favorites(data):
            return jsonify({'error': 'Failed to save favorites'}), 500

        logger.info(f"[FAVORITES] Successfully updated favorite: {run_id}")

        return jsonify({
            'success': True,
            'run_id': run_id
        }), 200

    except Exception as e:
        logger.error(f"[FAVORITES] Error updating favorite: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'Failed to update favorite',
            'details': str(e)
        }), 500


@favorites_bp.route('/check/<run_id>', methods=['GET'])
def check_favorite(run_id: str):
    """
    Check if a run is favorited.

    Args:
        run_id: Run ID to check

    Returns:
        200: Status of favorite

    Example:
        GET /api/favorites/check/run_abc123
    """
    try:
        data = _load_favorites()
        favorites = data.get('favorites', [])

        is_favorited = any(f.get('run_id') == run_id for f in favorites)

        return jsonify({
            'run_id': run_id,
            'is_favorited': is_favorited
        }), 200

    except Exception as e:
        logger.error(f"[FAVORITES] Error checking favorite: {e}")
        return jsonify({
            'error': 'Failed to check favorite',
            'details': str(e)
        }), 500
