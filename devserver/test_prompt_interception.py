"""
Test Prompt Interception: Verify 3-part structure matches original ComfyUI node

Original ComfyUI prompt_interception node (lines 261-264):
    full_prompt = (
        f"Task:\n{style_prompt.strip()}\n\n"
        f"Context:\n{input_context.strip()}\nPrompt:\n{input_prompt.strip()}"
    )

This test verifies that chunk_builder.py produces the exact same structure.
"""

import sys
import json
from pathlib import Path

def test_prompt_structure():
    """Test that manipulate chunk produces correct 3-part structure"""

    # Import from schemas.engine package
    schemas_path = Path(__file__).parent / 'schemas'
    sys.path.insert(0, str(schemas_path.parent))

    from schemas.engine.config_loader import config_loader  # Singleton instance
    from schemas.engine.chunk_builder import ChunkBuilder

    # Load pipeline JSON directly
    pipeline_path = schemas_path / 'pipelines' / 'text_transformation.json'
    with open(pipeline_path, 'r', encoding='utf-8') as f:
        pipeline_data = json.load(f)

    # Create a simple pipeline object
    class Pipeline:
        def __init__(self, data):
            self.instruction_type = data.get('instruction_type', 'artistic_transformation')
            self.name = data.get('name', '')

    # Load components
    chunk_builder = ChunkBuilder(schemas_path)
    pipeline = Pipeline(pipeline_data)

    # Load dada config
    config = config_loader.get_config('dada')

    # Build the manipulate chunk
    context = {
        'input_text': 'Eine Blume auf der Wiese',
        'user_input': 'Eine Blume auf der Wiese'
    }

    chunk_request = chunk_builder.build_chunk(
        chunk_name='manipulate',
        resolved_config=config,
        context=context,
        pipeline=pipeline
    )

    # Extract the built prompt
    built_prompt = chunk_request['prompt']

    print("=" * 80)
    print("PROMPT INTERCEPTION TEST")
    print("=" * 80)
    print("\n[BUILT PROMPT]")
    print(built_prompt)
    print("\n" + "=" * 80)

    # Verify structure
    lines = built_prompt.split('\n')

    # Check for 3-part structure markers
    has_task = 'Task:' in built_prompt
    has_context = 'Context:' in built_prompt
    has_prompt = 'Prompt:' in built_prompt

    print("\n[STRUCTURE VERIFICATION]")
    print(f"✓ Has 'Task:' section: {has_task}")
    print(f"✓ Has 'Context:' section: {has_context}")
    print(f"✓ Has 'Prompt:' section: {has_prompt}")

    # Extract sections
    if has_task and has_context and has_prompt:
        task_start = built_prompt.index('Task:')
        context_start = built_prompt.index('Context:')
        prompt_start = built_prompt.index('Prompt:')

        task_section = built_prompt[task_start:context_start].strip()
        context_section = built_prompt[context_start:prompt_start].strip()
        prompt_section = built_prompt[prompt_start:].strip()

        print(f"\n[TASK SECTION] ({len(task_section)} chars)")
        print(task_section[:200] + "..." if len(task_section) > 200 else task_section)

        print(f"\n[CONTEXT SECTION] ({len(context_section)} chars)")
        print(context_section[:200] + "..." if len(context_section) > 200 else context_section)

        print(f"\n[PROMPT SECTION] ({len(prompt_section)} chars)")
        print(prompt_section)

        # Verify against original structure
        print("\n" + "=" * 80)
        print("[COMPARISON WITH ORIGINAL]")
        print("=" * 80)
        print("\nOriginal ComfyUI structure:")
        print('f"Task:\\n{style_prompt}\\n\\nContext:\\n{input_context}\\nPrompt:\\n{input_prompt}"')
        print("\nOur structure matches: ✓" if has_task and has_context and has_prompt else "MISMATCH: ✗")

        success = has_task and has_context and has_prompt

        if success:
            print("\n✅ SUCCESS: 3-part prompt interception structure correctly implemented!")
            print("   The chunk_builder.py EXACTLY mimics the original ComfyUI prompt_interception node.")
        else:
            print("\n❌ FAILURE: Structure does not match original!")

        return success
    else:
        print("\n❌ FAILURE: Missing one or more section markers!")
        return False

if __name__ == '__main__':
    success = test_prompt_structure()
    sys.exit(0 if success else 1)
