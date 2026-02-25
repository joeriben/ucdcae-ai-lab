"""
Execution Tracker - Stateful execution history collector

The ExecutionTracker is the central component for tracking pipeline execution.
It collects events in-memory during execution, then persists after completion.

Design Principles:
- Request-scoped: One tracker per pipeline execution
- Non-blocking: <1ms per log call (in-memory only, no I/O)
- Fail-safe: All public methods wrapped in try-catch
- Explicit passing: Tracker passed as parameter (no globals)

Performance Target:
- <1ms per event logging
- <100ms total overhead for full pipeline
- No disk I/O during execution

Created: 2025-11-03 (Session 20 - Phase 1)
Based on: docs/EXECUTION_TRACKER_ARCHITECTURE.md Section 4
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
import time
import logging
import uuid

from .models import ExecutionItem, ExecutionRecord, MediaType, ItemType
from .storage import save_execution_record

logger = logging.getLogger(__name__)


class ExecutionTracker:
    """
    Stateful execution tracker - collects items during pipeline execution

    Usage:
        tracker = ExecutionTracker('dada', 'eco', 'kids')  # NOTE: 'eco' deprecated (Session 65)
        tracker.log_pipeline_start(input_text="Test", metadata={})
        tracker.log_user_input_text("Test input")
        # ... more logging calls ...
        tracker.finalize()  # Persist to disk

    Design: In-memory only during execution, persisted after completion
    Performance: <1ms per log_* call (list append only)
    Fail-safe: All public methods wrapped in try-catch
    """

    def __init__(
        self,
        config_name: str,
        safety_level: str,
        user_id: str = 'anonymous',
        session_id: str = 'default'
    ):
        """
        Initialize tracker for a single pipeline execution

        Args:
            config_name: Schema/config being executed (dada, stillepost, etc.)
            safety_level: kids | teens | adults
            user_id: User identifier
            session_id: Session identifier
        """
        self.execution_id = self._generate_execution_id()
        self.config_name = config_name
        self.safety_level = safety_level
        self.user_id = user_id
        self.session_id = session_id

        # State tracking
        self.current_stage = 0
        self.current_stage_iteration: Optional[int] = None
        self.current_loop_iteration: Optional[int] = None
        self.sequence_counter = 0

        # Collected items (in-memory during execution)
        self.items: List[ExecutionItem] = []

        # Timing
        self.start_time = time.time()
        self.finalized = False

        logger.info(f"[TRACKER] Created tracker {self.execution_id} for config '{config_name}'")

    def _generate_execution_id(self) -> str:
        """Generate unique execution ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique = uuid.uuid4().hex[:8]
        return f"exec_{timestamp}_{unique}"

    # ========================================================================
    # STATE MANAGEMENT (called by orchestrator)
    # ========================================================================

    def set_stage(self, stage: int):
        """Update current stage context"""
        self.current_stage = stage

    def set_stage_iteration(self, iteration: Optional[int]):
        """Update current stage iteration (for recursive pipelines like Stille Post)"""
        self.current_stage_iteration = iteration

    def set_loop_iteration(self, iteration: Optional[int]):
        """Update current loop iteration (for multi-output Stage 3-4)"""
        self.current_loop_iteration = iteration

    # ========================================================================
    # LOGGING METHODS (called by stage functions)
    # ========================================================================
    # All methods follow fail-safe pattern: try-catch, log warning, continue

    def log_pipeline_start(self, input_text: str, metadata: Dict[str, Any] = None):
        """Log pipeline start event"""
        try:
            self._log_item(
                stage=0,
                media_type=MediaType.METADATA,
                item_type=ItemType.PIPELINE_START,
                content=None,
                metadata={
                    'config_name': self.config_name,
                    'safety_level': self.safety_level,
                    'user_id': self.user_id,
                    'session_id': self.session_id,
                    **(metadata or {})
                }
            )
        except Exception as e:
            logger.warning(f"[TRACKER] Failed to log pipeline start: {e}")

    def log_user_input_text(self, text: str):
        """Log user text input (Stage 1)"""
        try:
            self._log_item(
                stage=1,
                media_type=MediaType.TEXT,
                item_type=ItemType.USER_INPUT_TEXT,
                content=text,
                metadata={
                    'input_language': 'de',  # Could be auto-detected
                    'char_count': len(text)
                }
            )
        except Exception as e:
            logger.warning(f"[TRACKER] Failed to log user input: {e}")

    def log_user_input_image(self, file_path: str, metadata: Dict[str, Any] = None):
        """Log user image input (Stage 1)"""
        try:
            self._log_item(
                stage=1,
                media_type=MediaType.IMAGE,
                item_type=ItemType.USER_INPUT_IMAGE,
                file_path=file_path,
                metadata=metadata or {}
            )
        except Exception as e:
            logger.warning(f"[TRACKER] Failed to log user image input: {e}")

    def log_translation_result(
        self,
        translated_text: str,
        from_lang: str,
        to_lang: str,
        model_used: str,
        execution_time: float,
        backend_type: str = None
    ):
        """Log translation result (Stage 1)"""
        try:
            self._log_item(
                stage=1,
                media_type=MediaType.TEXT,
                item_type=ItemType.TRANSLATION_RESULT,
                content=translated_text,
                model_used=model_used,
                backend_type=backend_type,
                execution_time=execution_time,
                metadata={
                    'from_lang': from_lang,
                    'to_lang': to_lang
                }
            )
        except Exception as e:
            logger.warning(f"[TRACKER] Failed to log translation: {e}")

    def log_stage1_safety_check(
        self,
        safe: bool,
        method: str,
        execution_time: float = 0.001,
        found_terms: List[str] = None,
        model_used: str = None
    ):
        """Log Stage 1 safety check result"""
        try:
            self._log_item(
                stage=1,
                media_type=MediaType.METADATA,
                item_type=ItemType.STAGE1_SAFETY_CHECK,
                content=None,
                model_used=model_used,
                execution_time=execution_time,
                metadata={
                    'safe': safe,
                    'method': method,
                    'found_terms': found_terms or [],
                    'risk_level': 'none' if safe else 'high'
                }
            )
        except Exception as e:
            logger.warning(f"[TRACKER] Failed to log stage1 safety check: {e}")

    def log_stage1_blocked(
        self,
        blocked_reason: str,
        blocked_codes: List[str],
        model_used: str = None,
        error_message: str = None
    ):
        """Log Stage 1 blocked event (pipeline stops here)"""
        try:
            self._log_item(
                stage=1,
                media_type=MediaType.METADATA,
                item_type=ItemType.STAGE1_BLOCKED,
                content=error_message,
                model_used=model_used,
                metadata={
                    'blocked_reason': blocked_reason,
                    'blocked_codes': blocked_codes
                }
            )
        except Exception as e:
            logger.warning(f"[TRACKER] Failed to log stage1 blocked: {e}")

    def log_interception_iteration(
        self,
        iteration_num: int,
        result_text: str,
        from_lang: str,
        to_lang: str,
        model_used: str,
        config_used: str = None
    ):
        """Log one interception iteration (Stage 2, recursive)"""
        try:
            self._log_item(
                stage=2,
                stage_iteration=iteration_num,
                media_type=MediaType.TEXT,
                item_type=ItemType.INTERCEPTION_ITERATION,
                content=result_text,
                config_used=config_used,
                model_used=model_used,
                metadata={
                    'from_lang': from_lang,
                    'to_lang': to_lang,
                    'iteration_type': 'translation'
                }
            )
        except Exception as e:
            logger.warning(f"[TRACKER] Failed to log interception iteration: {e}")

    def log_interception_final(
        self,
        final_text: str,
        total_iterations: int,
        config_used: str = None,
        model_used: str = None,
        backend_type: str = None,
        execution_time: float = None
    ):
        """Log final interception result (Stage 2)"""
        try:
            self._log_item(
                stage=2,
                media_type=MediaType.TEXT,
                item_type=ItemType.INTERCEPTION_FINAL,
                content=final_text,
                config_used=config_used,
                model_used=model_used,
                backend_type=backend_type,
                execution_time=execution_time,
                metadata={
                    'total_iterations': total_iterations,
                    'transformation_type': config_used or 'generic'
                }
            )
        except Exception as e:
            logger.warning(f"[TRACKER] Failed to log interception final: {e}")

    def log_stage3_safety_check(
        self,
        loop_iteration: int,
        safe: bool,
        method: str,
        config_used: str,
        model_used: str = None,
        backend_type: str = None,
        execution_time: float = None
    ):
        """Log Stage 3 pre-output safety check"""
        try:
            self._log_item(
                stage=3,
                loop_iteration=loop_iteration,
                media_type=MediaType.METADATA,
                item_type=ItemType.STAGE3_SAFETY_CHECK,
                content=None,
                config_used=config_used,
                model_used=model_used,
                backend_type=backend_type,
                execution_time=execution_time,
                metadata={
                    'safe': safe,
                    'method': method,
                    'safety_level': self.safety_level
                }
            )
        except Exception as e:
            logger.warning(f"[TRACKER] Failed to log stage3 safety check: {e}")

    def log_stage3_blocked(
        self,
        loop_iteration: int,
        config_used: str,
        abort_reason: str
    ):
        """Log Stage 3 blocked event (output skipped, not entire pipeline)"""
        try:
            self._log_item(
                stage=3,
                loop_iteration=loop_iteration,
                media_type=MediaType.METADATA,
                item_type=ItemType.STAGE3_BLOCKED,
                content=abort_reason,
                config_used=config_used,
                metadata={
                    'blocked_reason': 'age_inappropriate',
                    'safety_level': self.safety_level,
                    'fallback': 'text_alternative_provided'
                }
            )
        except Exception as e:
            logger.warning(f"[TRACKER] Failed to log stage3 blocked: {e}")

    def log_output_image(
        self,
        loop_iteration: int,
        config_used: str,
        file_path: str,
        model_used: str,
        backend_type: str,
        metadata: Dict[str, Any],
        execution_time: float = None
    ):
        """Log Stage 4 image output"""
        try:
            self._log_item(
                stage=4,
                loop_iteration=loop_iteration,
                media_type=MediaType.IMAGE,
                item_type=ItemType.OUTPUT_IMAGE,
                file_path=file_path,
                config_used=config_used,
                model_used=model_used,
                backend_type=backend_type,
                execution_time=execution_time,
                metadata=metadata  # width, height, seed, cfg_scale, steps, sampler, etc.
            )
        except Exception as e:
            logger.warning(f"[TRACKER] Failed to log output image: {e}")

    def log_output_audio(
        self,
        loop_iteration: int,
        config_used: str,
        file_path: str,
        model_used: str,
        backend_type: str,
        metadata: Dict[str, Any],
        execution_time: float = None
    ):
        """Log Stage 4 audio output"""
        try:
            self._log_item(
                stage=4,
                loop_iteration=loop_iteration,
                media_type=MediaType.AUDIO,
                item_type=ItemType.OUTPUT_AUDIO,
                file_path=file_path,
                config_used=config_used,
                model_used=model_used,
                backend_type=backend_type,
                execution_time=execution_time,
                metadata=metadata  # duration, sample_rate, channels, format
            )
        except Exception as e:
            logger.warning(f"[TRACKER] Failed to log output audio: {e}")

    def log_output_music(
        self,
        loop_iteration: int,
        config_used: str,
        file_path: str,
        model_used: str,
        backend_type: str,
        metadata: Dict[str, Any],
        execution_time: float = None
    ):
        """Log Stage 4 music output"""
        try:
            self._log_item(
                stage=4,
                loop_iteration=loop_iteration,
                media_type=MediaType.MUSIC,
                item_type=ItemType.OUTPUT_MUSIC,
                file_path=file_path,
                config_used=config_used,
                model_used=model_used,
                backend_type=backend_type,
                execution_time=execution_time,
                metadata=metadata  # duration, genre, tempo, key, format
            )
        except Exception as e:
            logger.warning(f"[TRACKER] Failed to log output music: {e}")

    def log_pipeline_complete(
        self,
        total_duration: float,
        outputs_generated: int
    ):
        """Log pipeline completion event"""
        try:
            # Temporarily clear loop_iteration for pipeline_complete (not part of any loop)
            saved_loop_iteration = self.current_loop_iteration
            self.current_loop_iteration = None

            self._log_item(
                stage=5,
                media_type=MediaType.METADATA,
                item_type=ItemType.PIPELINE_COMPLETE,
                content=None,
                metadata={
                    'total_duration_seconds': total_duration,
                    'total_items_logged': len(self.items) + 1,  # +1 for this item
                    'outputs_generated': outputs_generated,
                    'stages_completed': [1, 2, 3, 4]
                }
            )

            # Restore loop_iteration
            self.current_loop_iteration = saved_loop_iteration
        except Exception as e:
            logger.warning(f"[TRACKER] Failed to log pipeline complete: {e}")

    def log_pipeline_error(
        self,
        error_type: str,
        error_message: str,
        stage: int
    ):
        """Log pipeline error event"""
        try:
            self._log_item(
                stage=stage,
                media_type=MediaType.METADATA,
                item_type=ItemType.PIPELINE_ERROR,
                content=f"{error_type}: {error_message}",
                metadata={
                    'error_type': error_type,
                    'error_stage': stage,
                    'error_message': error_message,
                    'recoverable': False
                }
            )
        except Exception as e:
            logger.warning(f"[TRACKER] Failed to log pipeline error: {e}")

    def log_stage_transition(self, from_stage: int, to_stage: int):
        """Log stage transition event"""
        try:
            self._log_item(
                stage=from_stage,  # Current stage before transition
                media_type=MediaType.METADATA,
                item_type=ItemType.STAGE_TRANSITION,
                content=None,
                metadata={
                    'from_stage': from_stage,
                    'to_stage': to_stage,
                    'transition_time_ms': 5  # Negligible
                }
            )
        except Exception as e:
            logger.warning(f"[TRACKER] Failed to log stage transition: {e}")

    # ========================================================================
    # INTERNAL HELPERS
    # ========================================================================

    def _log_item(
        self,
        stage: int,
        media_type: MediaType,
        item_type: ItemType,
        content: Optional[str] = None,
        file_path: Optional[str] = None,
        config_used: Optional[str] = None,
        model_used: Optional[str] = None,
        backend_type: Optional[str] = None,
        execution_time: Optional[float] = None,
        stage_iteration: Optional[int] = None,
        loop_iteration: Optional[int] = None,
        metadata: Dict[str, Any] = None
    ):
        """
        Internal method to log an item (fast in-memory append)

        Performance: ~0.1-0.5ms (list append + dataclass creation)
        """
        self.sequence_counter += 1

        item = ExecutionItem(
            sequence_number=self.sequence_counter,
            timestamp=datetime.now(),
            stage=stage,
            stage_iteration=stage_iteration or self.current_stage_iteration,
            loop_iteration=loop_iteration or self.current_loop_iteration,
            media_type=media_type,
            item_type=item_type,
            content=content,
            file_path=file_path,
            config_used=config_used,
            model_used=model_used,
            backend_type=backend_type,
            execution_time=execution_time,
            metadata=metadata or {}
        )

        self.items.append(item)  # FAST: in-memory list append

    def finalize(self):
        """
        Persist execution record to storage (called AFTER pipeline completes)

        This is the ONLY method that does disk I/O
        """
        if self.finalized:
            logger.warning(f"[TRACKER] Tracker {self.execution_id} already finalized")
            return

        self.finalized = True

        try:
            # Build complete record
            record = ExecutionRecord(
                execution_id=self.execution_id,
                config_name=self.config_name,
                timestamp=datetime.fromtimestamp(self.start_time),
                user_id=self.user_id,
                session_id=self.session_id,

                safety_level=self.safety_level,
                used_seed=None,  # TODO: Extract from media outputs if needed
                total_execution_time=time.time() - self.start_time,
                items=self.items,
                taxonomy_version="1.0"
            )

            # Persist to storage
            save_execution_record(record)

            logger.info(
                f"[TRACKER] Finalized {self.execution_id}: "
                f"{len(self.items)} items, "
                f"{record.total_execution_time:.1f}s total"
            )

        except Exception as e:
            logger.error(f"[TRACKER] Failed to finalize tracker {self.execution_id}: {e}")
            # Don't raise - tracker failures should never break pipeline

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def get_execution_record(self) -> ExecutionRecord:
        """Get current execution record (without finalizing)"""
        return ExecutionRecord(
            execution_id=self.execution_id,
            config_name=self.config_name,
            timestamp=datetime.fromtimestamp(self.start_time),
            user_id=self.user_id,
            session_id=self.session_id,
            safety_level=self.safety_level,
            used_seed=None,
            total_execution_time=time.time() - self.start_time,
            items=self.items,
            taxonomy_version="1.0"
        )

    def get_items_by_stage(self, stage: int) -> List[ExecutionItem]:
        """Get all items for a specific stage"""
        return [item for item in self.items if item.stage == stage]

    def get_items_by_type(self, item_type: ItemType) -> List[ExecutionItem]:
        """Get all items of a specific type"""
        return [item for item in self.items if item.item_type == item_type]
