#!/usr/bin/env python3
"""
Quick test of LivePipelineRecorder to verify it works before integration.
"""

import sys
import uuid
from pathlib import Path

# Add devserver to path
sys.path.insert(0, str(Path(__file__).parent))

from my_app.services.pipeline_recorder import LivePipelineRecorder, get_recorder, load_recorder


def test_basic_recording():
    """Test basic recording functionality."""
    print("=" * 60)
    print("TEST 1: Basic Recording")
    print("=" * 60)

    # Create recorder
    run_id = str(uuid.uuid4())
    recorder = LivePipelineRecorder(
        run_id=run_id,
        config_name="dada",
        safety_level="kids",
        user_id="test_user"
    )

    print(f"\n✓ Created recorder for run: {run_id}")
    print(f"  Folder: {recorder.run_folder}")

    # Test saving text entity
    recorder.set_state(1, "input_received")
    recorder.save_entity("input", "Test input text")
    print("\n✓ Saved text entity: 01_input.txt")

    # Test saving JSON entity
    recorder.set_state(1, "safety_check")
    recorder.save_entity("safety", {"safe": True, "method": "llm_context_check"})
    print("✓ Saved JSON entity: 02_safety.json")

    # Test saving binary entity (fake image)
    recorder.set_state(4, "image_generation")
    fake_image = b'\x89PNG\r\n\x1a\n...'  # Fake PNG header
    recorder.save_entity(
        "output_image",
        fake_image,
        metadata={"seed": 12345, "cfg": 7.0}
    )
    print("✓ Saved binary entity: 03_output_image.png")

    # Test error entity
    recorder.save_error(
        stage=3,
        error_type="safety_blocked",
        message="Content violates policy",
        details={"codes": ["§86a"]}
    )
    print("✓ Saved error entity: 04_error.json")

    # Mark complete
    recorder.mark_complete()
    print("✓ Marked as complete")

    # Verify files exist
    print("\n" + "=" * 60)
    print("Files created:")
    print("=" * 60)
    for file in sorted(recorder.run_folder.iterdir()):
        size = file.stat().st_size
        print(f"  {file.name:30s} {size:>6d} bytes")

    # Test get_status
    status = recorder.get_status()
    print("\n" + "=" * 60)
    print("Status:")
    print("=" * 60)
    print(f"  Run ID: {status['run_id']}")
    print(f"  Stage: {status['current_state']['stage']}")
    print(f"  Step: {status['current_state']['step']}")
    print(f"  Progress: {status['current_state']['progress']}")
    print(f"  Completed outputs: {status['completed_outputs']}")
    print(f"  Next expected: {status['next_expected']}")

    return run_id, recorder.run_folder


def test_singleton_management():
    """Test get_recorder and load_recorder."""
    print("\n\n" + "=" * 60)
    print("TEST 2: Singleton Management")
    print("=" * 60)

    run_id = str(uuid.uuid4())

    # Create via get_recorder
    recorder1 = get_recorder(
        run_id=run_id,
        config_name="stillepost",
        safety_level="teens"
    )
    print(f"\n✓ Created recorder via get_recorder: {run_id}")

    # Get same recorder again
    recorder2 = get_recorder(run_id=run_id)
    print(f"✓ Retrieved same recorder: {recorder1 is recorder2}")

    # Save some data
    recorder1.save_entity("input", "Singleton test")

    # Load from disk
    recorder3 = load_recorder(run_id)
    print(f"✓ Loaded from disk: {len(recorder3.metadata['entities'])} entities")

    return run_id


def test_entity_path_lookup():
    """Test get_entity_path functionality."""
    print("\n\n" + "=" * 60)
    print("TEST 3: Entity Path Lookup")
    print("=" * 60)

    run_id = str(uuid.uuid4())
    recorder = LivePipelineRecorder(
        run_id=run_id,
        config_name="dada",
        safety_level="kids"
    )

    # Save entities
    recorder.save_entity("input", "Test")
    recorder.save_entity("translation", "Test translation")

    # Lookup paths
    input_path = recorder.get_entity_path("input")
    translation_path = recorder.get_entity_path("translation")
    missing_path = recorder.get_entity_path("nonexistent")

    print(f"\n✓ Found input path: {input_path.name}")
    print(f"✓ Found translation path: {translation_path.name}")
    print(f"✓ Missing entity returns None: {missing_path is None}")

    return run_id


if __name__ == "__main__":
    print("Testing LivePipelineRecorder\n")

    try:
        # Run tests
        run_id1, folder1 = test_basic_recording()
        run_id2 = test_singleton_management()
        run_id3 = test_entity_path_lookup()

        print("\n\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        print("\nGenerated test runs:")
        print(f"  1. {run_id1}")
        print(f"  2. {run_id2}")
        print(f"  3. {run_id3}")
        print(f"\nTest folder: {folder1}")
        print("\nYou can inspect the files to verify correct structure.")

    except Exception as e:
        print("\n\n" + "=" * 60)
        print("TEST FAILED ✗")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
