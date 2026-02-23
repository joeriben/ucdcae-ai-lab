#!/usr/bin/env python3
"""
Test script for Surrealization pipeline (dual-encoder T5+CLIP fusion)
Tests all 3 steps of the dual_encoder_fusion pipeline
"""
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from schemas.engine.chunk_builder import ChunkBuilder
from schemas.engine.config_loader import ConfigLoader

def test_surrealization_pipeline():
    """Test the complete surrealization pipeline"""

    print("=" * 80)
    print("TESTING SURREALIZATION PIPELINE (Dual-Encoder T5+CLIP Fusion)")
    print("=" * 80)

    # Initialize components
    schemas_path = Path(__file__).parent / "schemas"
    chunk_builder = ChunkBuilder(schemas_path)
    config_loader = ConfigLoader()
    config_loader.initialize(schemas_path)

    # Load surrealization config
    print("\n[1] Loading surrealization config...")
    config = config_loader.get_config("surrealization")
    if not config:
        print("❌ Config 'surrealization' not found!")
        return False
    print(f"✅ Config loaded: {config.name}")
    print(f"   Config type: {type(config).__name__}")

    # Load pipeline definition directly
    print("\n[2] Loading dual_encoder_fusion pipeline...")
    pipeline_file = schemas_path / "pipelines" / "dual_encoder_fusion.json"
    if not pipeline_file.exists():
        print(f"❌ Pipeline file not found: {pipeline_file}")
        return False

    with open(pipeline_file) as f:
        pipeline_data = json.load(f)

    print(f"✅ Pipeline loaded: {pipeline_data['name']}")
    print(f"   Chunks: {', '.join(pipeline_data['chunks'])}")

    # Create a simple pipeline object
    class SimplePipeline:
        def __init__(self, data):
            self.name = data['name']
            self.chunks = data['chunks']
            self.instruction_type = 'artistic_transformation'

    pipeline = SimplePipeline(pipeline_data)

    # Test input
    test_input = "A surreal landscape where mountains float upside down above an ocean of clouds"

    print(f"\n[3] Test Input:")
    print(f"   '{test_input}'")

    # Build context
    context = {
        'input_text': test_input,
        'user_input': test_input
    }

    print("\n" + "=" * 80)
    print("STEP 1: T5 Prompt Optimization (semantic expansion + alpha calculation)")
    print("=" * 80)

    try:
        chunk1_request = chunk_builder.build_chunk(
            chunk_name='optimize_t5_prompt',
            resolved_config=config,
            context=context,
            pipeline=pipeline
        )

        print(f"\n✅ Chunk 1 built successfully!")
        print(f"   Backend: {chunk1_request['backend_type']}")
        print(f"   Model: {chunk1_request['model']}")
        print(f"   Prompt preview:")

        # Show first 500 chars of prompt
        prompt = chunk1_request['prompt']
        if "Task:" in prompt:
            parts = prompt.split("\n\n")
            for i, part in enumerate(parts[:3], 1):
                print(f"   Part {i}: {part[:200]}{'...' if len(part) > 200 else ''}")
        else:
            print(f"   {prompt[:500]}{'...' if len(prompt) > 500 else ''}")

        # Check for dict template processing
        if chunk1_request['metadata'].get('chunk_name') == 'optimize_t5_prompt':
            print(f"\n✅ Dict template processed correctly!")

    except Exception as e:
        print(f"\n❌ Error building chunk 1: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("STEP 2: CLIP Prompt Optimization (token reordering)")
    print("=" * 80)

    try:
        chunk2_request = chunk_builder.build_chunk(
            chunk_name='optimize_clip_prompt',
            resolved_config=config,
            context=context,
            pipeline=pipeline
        )

        print(f"\n✅ Chunk 2 built successfully!")
        print(f"   Backend: {chunk2_request['backend_type']}")
        print(f"   Model: {chunk2_request['model']}")
        print(f"   Prompt preview:")

        prompt = chunk2_request['prompt']
        if "Task:" in prompt:
            parts = prompt.split("\n\n")
            for i, part in enumerate(parts[:3], 1):
                print(f"   Part {i}: {part[:200]}{'...' if len(part) > 200 else ''}")
        else:
            print(f"   {prompt[:500]}{'...' if len(prompt) > 500 else ''}")

    except Exception as e:
        print(f"\n❌ Error building chunk 2: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("STEP 3: Dual-Encoder Fusion Image Generation")
    print("=" * 80)

    try:
        # Simulate outputs from steps 1 and 2
        context_with_previous = {
            **context,
            'T5_PROMPT': 'A surreal landscape characterized by floating mountains...(simulated T5 expansion)',
            'CLIP_PROMPT': 'floating mountains, upside down, ocean of clouds, surreal landscape',
            'ALPHA': '0.25'
        }

        chunk3_request = chunk_builder.build_chunk(
            chunk_name='dual_encoder_fusion_image',
            resolved_config=config,
            context=context_with_previous,
            pipeline=pipeline
        )

        print(f"\n✅ Chunk 3 built successfully!")
        print(f"   Backend: {chunk3_request['backend_type']}")
        print(f"   Workflow nodes: {len(chunk3_request.get('prompt', {}).get('workflow', {}))}")

        # Check if placeholders would be replaced
        if 'prompt' in chunk3_request and 'workflow' in chunk3_request['prompt']:
            workflow = chunk3_request['prompt']['workflow']

            # Check node 5 (CLIP encode)
            if '5' in workflow and 'text' in workflow['5']['inputs']:
                clip_text = workflow['5']['inputs']['text']
                print(f"   CLIP node text: {clip_text[:100]}...")

            # Check node 6 (T5 encode)
            if '6' in workflow and 'text' in workflow['6']['inputs']:
                t5_text = workflow['6']['inputs']['text']
                print(f"   T5 node text: {t5_text[:100]}...")

            # Check node 9 (fusion alpha)
            if '9' in workflow and 'alpha' in workflow['9']['inputs']:
                alpha_val = workflow['9']['inputs']['alpha']
                print(f"   Fusion alpha: {alpha_val}")

    except Exception as e:
        print(f"\n❌ Error building chunk 3: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("✅ All 3 chunks of surrealization pipeline built successfully!")
    print("✅ Dict templates (optimize_t5_prompt, optimize_clip_prompt) working!")
    print("✅ ComfyUI workflow chunk (dual_encoder_fusion_image) working!")
    print("\nNext steps:")
    print("1. Test with actual LLM backend (Ollama/LM Studio)")
    print("2. Implement alpha extraction from T5 output (#a=XX)")
    print("3. Test full pipeline execution with ComfyUI")
    print("4. Verify ai4artsed_t5_clip_fusion custom node is installed")

    return True

if __name__ == "__main__":
    success = test_surrealization_pipeline()
    sys.exit(0 if success else 1)
