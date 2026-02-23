"""
Flask routes for execution history API

Provides REST endpoints for querying and exporting execution records.

Phase 3: Export API (Session 22)
Based on: docs/SESSION_21_HANDOVER.md, docs/EXECUTION_TRACKER_ARCHITECTURE.md

Created: 2025-11-03 (Session 22 - Phase 3)
"""
import logging
from flask import Blueprint, jsonify, request, Response
from datetime import datetime
from typing import Optional

from execution_history.storage import (
    load_execution_record,
    list_execution_records,
    get_storage_stats
)

logger = logging.getLogger(__name__)

# Create blueprint
execution_bp = Blueprint('execution', __name__)


@execution_bp.route('/api/runs/<execution_id>', methods=['GET'])
def get_execution(execution_id: str):
    """
    Get single execution record by ID

    Returns:
        200: ExecutionRecord as JSON
        404: Execution not found
        500: Server error

    Example:
        GET /api/runs/exec_20251103_205239_896e054c
    """
    try:
        logger.info(f"[EXECUTION_API] Fetching execution: {execution_id}")

        # Load execution record
        record = load_execution_record(execution_id)

        if not record:
            logger.warning(f"[EXECUTION_API] Execution not found: {execution_id}")
            return jsonify({
                'error': 'Execution not found',
                'execution_id': execution_id
            }), 404

        # Convert to dict for JSON response
        record_dict = record.to_dict()

        logger.info(f"[EXECUTION_API] Successfully fetched execution: {execution_id} ({len(record.items)} items)")
        return jsonify(record_dict), 200

    except Exception as e:
        logger.error(f"[EXECUTION_API] Error fetching execution {execution_id}: {e}")
        return jsonify({
            'error': 'Failed to fetch execution',
            'execution_id': execution_id,
            'details': str(e)
        }), 500


@execution_bp.route('/api/runs', methods=['GET'])
def list_executions():
    """
    List execution records with optional filtering and pagination

    Query Parameters:
        - limit: Max number of results (default: 20, max: 100)
        - offset: Number of results to skip (default: 0)
        - config: Filter by config name (e.g., "dada", "stillepost")
        - date: Filter by date (format: YYYY-MM-DD)
        - user_id: Filter by user ID
        - session_id: Filter by session ID

    Returns:
        200: List of execution summaries
        400: Invalid parameters
        500: Server error

    Example:
        GET /api/runs?limit=20&offset=0&config=dada&date=2025-11-03
    """
    try:
        # Parse query parameters
        limit = request.args.get('limit', 20, type=int)
        offset = request.args.get('offset', 0, type=int)
        config_filter = request.args.get('config', None, type=str)
        date_filter = request.args.get('date', None, type=str)
        user_id_filter = request.args.get('user_id', None, type=str)
        session_id_filter = request.args.get('session_id', None, type=str)

        # Validate parameters
        if limit < 1 or limit > 100:
            return jsonify({
                'error': 'Invalid limit parameter',
                'details': 'Limit must be between 1 and 100'
            }), 400

        if offset < 0:
            return jsonify({
                'error': 'Invalid offset parameter',
                'details': 'Offset must be >= 0'
            }), 400

        # Validate date format if provided
        if date_filter:
            try:
                datetime.strptime(date_filter, '%Y-%m-%d')
            except ValueError:
                return jsonify({
                    'error': 'Invalid date format',
                    'details': 'Date must be in format YYYY-MM-DD'
                }), 400

        logger.info(f"[EXECUTION_API] Listing executions: limit={limit}, offset={offset}, config={config_filter}, date={date_filter}")

        # Get all execution IDs (we'll filter them ourselves)
        # Note: This loads more than needed for filtering, but keeps storage.py simple
        # Future optimization: Add filtering to storage.py
        all_execution_ids = list_execution_records(limit=1000, offset=0)

        # Filter execution IDs by loading metadata
        filtered_executions = []

        for exec_id in all_execution_ids:
            # Load record to check filters
            record = load_execution_record(exec_id)

            if not record:
                continue

            # Apply filters
            if config_filter and record.config_name != config_filter:
                continue

            if date_filter:
                record_date = record.timestamp.strftime('%Y-%m-%d')
                if record_date != date_filter:
                    continue

            if user_id_filter and record.user_id != user_id_filter:
                continue

            if session_id_filter and record.session_id != session_id_filter:
                continue

            # Build summary (don't include all items for performance)
            execution_summary = {
                'execution_id': record.execution_id,
                'config_name': record.config_name,
                'timestamp': record.timestamp.isoformat(),
                'safety_level': record.safety_level,
                'user_id': record.user_id,
                'session_id': record.session_id,
                'total_execution_time': record.total_execution_time,
                'items_count': len(record.items),
                'taxonomy_version': record.taxonomy_version
            }

            filtered_executions.append(execution_summary)

        # Apply pagination to filtered results
        total_count = len(filtered_executions)
        paginated_executions = filtered_executions[offset:offset+limit]

        response = {
            'executions': paginated_executions,
            'total': total_count,
            'limit': limit,
            'offset': offset,
            'returned': len(paginated_executions)
        }

        # Include applied filters in response
        if config_filter or date_filter or user_id_filter or session_id_filter:
            response['filters'] = {
                'config': config_filter,
                'date': date_filter,
                'user_id': user_id_filter,
                'session_id': session_id_filter
            }

        logger.info(f"[EXECUTION_API] Successfully listed {len(paginated_executions)} executions (total: {total_count})")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"[EXECUTION_API] Error listing executions: {e}")
        return jsonify({
            'error': 'Failed to list executions',
            'details': str(e)
        }), 500


@execution_bp.route('/api/runs/<execution_id>/export/<format_type>', methods=['GET'])
def export_execution(execution_id: str, format_type: str):
    """
    Export execution record in specified format

    Supported formats:
        - json: Raw JSON format (same as GET /api/runs/<id>)
        - xml: Legacy XML format (future)
        - pdf: PDF report (future)
        - docx: DOCX report (future, legacy compatibility)

    Returns:
        200: Exported file
        400: Invalid format
        404: Execution not found
        501: Format not implemented yet
        500: Server error

    Example:
        GET /api/runs/exec_20251103_205239_896e054c/export/json
    """
    try:
        logger.info(f"[EXECUTION_API] Exporting execution {execution_id} as {format_type}")

        # Load execution record
        record = load_execution_record(execution_id)

        if not record:
            logger.warning(f"[EXECUTION_API] Execution not found for export: {execution_id}")
            return jsonify({
                'error': 'Execution not found',
                'execution_id': execution_id
            }), 404

        # Handle different export formats
        if format_type == 'json':
            # JSON export (same as GET endpoint, but with download headers)
            record_dict = record.to_dict()

            response = Response(
                response=jsonify(record_dict).get_data(as_text=True),
                status=200,
                mimetype='application/json',
                headers={
                    'Content-Disposition': f'attachment; filename="{execution_id}.json"'
                }
            )

            logger.info(f"[EXECUTION_API] Successfully exported {execution_id} as JSON")
            return response

        elif format_type == 'xml':
            # XML export (future implementation)
            logger.warning(f"[EXECUTION_API] XML export not yet implemented")
            return jsonify({
                'error': 'XML export not yet implemented',
                'status': 'planned',
                'details': 'This format will be available in a future release'
            }), 501

        elif format_type == 'pdf':
            # PDF export (future implementation)
            logger.warning(f"[EXECUTION_API] PDF export not yet implemented")
            return jsonify({
                'error': 'PDF export not yet implemented',
                'status': 'planned',
                'details': 'This format will be available in a future release'
            }), 501

        elif format_type == 'docx':
            # DOCX export (future implementation for legacy compatibility)
            logger.warning(f"[EXECUTION_API] DOCX export not yet implemented")
            return jsonify({
                'error': 'DOCX export not yet implemented',
                'status': 'planned',
                'details': 'This format will be available in a future release'
            }), 501

        else:
            # Invalid format
            logger.warning(f"[EXECUTION_API] Invalid export format: {format_type}")
            return jsonify({
                'error': 'Invalid export format',
                'format': format_type,
                'supported_formats': ['json', 'xml', 'pdf', 'docx'],
                'available_now': ['json']
            }), 400

    except Exception as e:
        logger.error(f"[EXECUTION_API] Error exporting execution {execution_id} as {format_type}: {e}")
        return jsonify({
            'error': 'Failed to export execution',
            'execution_id': execution_id,
            'format': format_type,
            'details': str(e)
        }), 500


@execution_bp.route('/api/runs/stats', methods=['GET'])
def get_execution_stats():
    """
    Get execution history storage statistics

    Returns:
        200: Storage stats (total records, disk usage, etc.)
        500: Server error

    Example:
        GET /api/runs/stats
    """
    try:
        logger.info(f"[EXECUTION_API] Fetching storage stats")

        stats = get_storage_stats()

        logger.info(f"[EXECUTION_API] Successfully fetched stats: {stats.get('total_records', 0)} records")
        return jsonify(stats), 200

    except Exception as e:
        logger.error(f"[EXECUTION_API] Error fetching stats: {e}")
        return jsonify({
            'error': 'Failed to fetch storage stats',
            'details': str(e)
        }), 500
