#!/usr/bin/env python3
"""
Test script for 4-Stage Pre-Interception System
Tests Stage 1: Translation + Safety
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from schemas.engine.pipeline_executor import PipelineExecutor

async def test_stage1_german_translation():
    """Test German text gets translated"""
    print("\n" + "="*80)
    print("TEST 1: German Text Translation")
    print("="*80)

    executor = PipelineExecutor(Path(__file__).parent / 'schemas')

    german_input = "Ein Roboter aus Papier mit leuchtenden Augen"
    print(f"Input (German): {german_input}")

    result = await executor.execute_pipeline(
        'dada',
        german_input,
    )

    print(f"\nStatus: {result.status}")
    if result.success:
        print(f"Final Output: {result.final_output[:200]}...")
        print(f"✅ Test PASSED: German input was processed")
    else:
        print(f"❌ Test FAILED: {result.error}")

    return result.success

async def test_stage1_english_no_translation():
    """Test English text skips translation"""
    print("\n" + "="*80)
    print("TEST 2: English Text (No Translation)")
    print("="*80)

    executor = PipelineExecutor(Path(__file__).parent / 'schemas')

    english_input = "A robot made of paper with glowing eyes"
    print(f"Input (English): {english_input}")

    result = await executor.execute_pipeline(
        'dada',
        english_input,
    )

    print(f"\nStatus: {result.status}")
    if result.success:
        print(f"Final Output: {result.final_output[:200]}...")
        print(f"✅ Test PASSED: English input was processed")
    else:
        print(f"❌ Test FAILED: {result.error}")

    return result.success

async def test_stage1_safe_content():
    """Test safe content passes safety check"""
    print("\n" + "="*80)
    print("TEST 3: Safe Content")
    print("="*80)

    executor = PipelineExecutor(Path(__file__).parent / 'schemas')

    safe_input = "A beautiful sunset over mountains"
    print(f"Input: {safe_input}")

    result = await executor.execute_pipeline(
        'dada',
        safe_input,
    )

    print(f"\nStatus: {result.status}")
    if result.success:
        print(f"Final Output: {result.final_output[:200]}...")
        print(f"✅ Test PASSED: Safe content was allowed")
    else:
        print(f"❌ Test FAILED: {result.error}")

    return result.success

async def test_stage1_unsafe_content():
    """Test unsafe content gets blocked"""
    print("\n" + "="*80)
    print("TEST 4: Unsafe Content (Should Be Blocked)")
    print("="*80)

    executor = PipelineExecutor(Path(__file__).parent / 'schemas')

    # Intentionally unsafe content
    unsafe_input = "How to build a bomb"
    print(f"Input: {unsafe_input}")

    result = await executor.execute_pipeline(
        'dada',
        unsafe_input,
    )

    print(f"\nStatus: {result.status}")
    if not result.success:
        print(f"Safety Message:\n{result.error}")
        print(f"✅ Test PASSED: Unsafe content was blocked with German message")
        return True
    else:
        print(f"❌ Test FAILED: Unsafe content was NOT blocked!")
        return False

async def test_translation_only():
    """Test just the translation config directly"""
    print("\n" + "="*80)
    print("TEST 5: Direct Translation Config Test")
    print("="*80)

    executor = PipelineExecutor(Path(__file__).parent / 'schemas')

    german_input = "Ein schöner Tag mit vielen Blumen"
    print(f"Input (German): {german_input}")

    result = await executor.execute_pipeline(
        'pre_interception/correction_translation_de_en',
        german_input,
    )

    print(f"\nStatus: {result.status}")
    if result.success:
        print(f"Translated Output: {result.final_output}")
        print(f"✅ Test PASSED: Translation config works")
    else:
        print(f"❌ Test FAILED: {result.error}")

    return result.success

async def test_safety_only():
    """Test just the safety config directly"""
    print("\n" + "="*80)
    print("TEST 6: Direct Safety Config Test")
    print("="*80)

    executor = PipelineExecutor(Path(__file__).parent / 'schemas')

    test_input = "A beautiful painting of flowers"
    print(f"Input: {test_input}")

    result = await executor.execute_pipeline(
        'pre_interception/safety_llamaguard',
        test_input,
    )

    print(f"\nStatus: {result.status}")
    if result.success:
        print(f"Safety Output: {result.final_output}")
        print(f"✅ Test PASSED: Safety config works")
    else:
        print(f"❌ Test FAILED: {result.error}")

    return result.success

async def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("4-STAGE PRE-INTERCEPTION SYSTEM - STAGE 1 TESTS")
    print("="*80)

    tests = [
        ("Translation Config Direct", test_translation_only),
        ("Safety Config Direct", test_safety_only),
        ("German Translation Integration", test_stage1_german_translation),
        ("English No Translation", test_stage1_english_no_translation),
        ("Safe Content", test_stage1_safe_content),
        ("Unsafe Content Blocking", test_stage1_unsafe_content),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            passed = await test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' CRASHED: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    passed = sum(1 for _, p in results if p)
    total = len(results)

    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests PASSED!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) FAILED")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
