"""
Execution History Package

Tracks and persists pipeline execution history for research and analysis.

Public API:
- ExecutionTracker: Main class for tracking execution
- ExecutionRecord: Complete execution history record
- ExecutionItem: Single tracked event
- MediaType: Media type enum
- ItemType: Item type enum
- Storage functions: save_execution_record, load_execution_record, list_execution_records

Usage:
    from execution_history import ExecutionTracker

    tracker = ExecutionTracker('dada', 'eco', 'kids')  # NOTE: 'eco' deprecated (Session 65)
    tracker.log_pipeline_start(input_text="Test", metadata={})
    tracker.log_user_input_text("Test")
    # ... more logging ...
    tracker.finalize()  # Persist to disk

Created: 2025-11-03 (Session 20 - Phase 1)
"""

from .models import (
    ExecutionItem,
    ExecutionRecord,
    MediaType,
    ItemType
)

from .tracker import ExecutionTracker

from .storage import (
    save_execution_record,
    load_execution_record,
    list_execution_records,
    delete_execution_record,
    get_storage_stats
)

__all__ = [
    # Core classes
    'ExecutionTracker',
    'ExecutionRecord',
    'ExecutionItem',

    # Enums
    'MediaType',
    'ItemType',

    # Storage functions
    'save_execution_record',
    'load_execution_record',
    'list_execution_records',
    'delete_execution_record',
    'get_storage_stats',
]

__version__ = '1.0.0'
