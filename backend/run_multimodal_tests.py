#!/usr/bin/env python3
"""
Comprehensive test runner for Multimodal RAG features.
Runs unit tests, edge case tests, and generates test report.
"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_command(cmd, description):
    """Run a command and capture output."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0, result.stdout
    
    except subprocess.TimeoutExpired:
        print(f"❌ Test timed out after 300 seconds")
        return False, ""
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False, ""


def main():
    """Run all multimodal tests."""
    print(f"""
================================================================================
                 MULTIMODAL RAG TEST SUITE                           
                                                                      
  Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                                    
================================================================================
""")
    
    backend_dir = Path(__file__).parent
    
    test_suites = [
        {
            "name": "Core Multimodal Tests",
            "cmd": f'python -m pytest tests/test_multimodal.py -v --tb=short -m "not integration"',
            "description": "Core unit tests for multimodal components"
        },
        {
            "name": "Edge Case Tests",
            "cmd": f'python -m pytest tests/test_multimodal_edge_cases.py -v --tb=short -m "not integration"',
            "description": "Edge cases and boundary conditions"
        },
        {
            "name": "Comprehensive Tests",
            "cmd": f'python -m pytest tests/test_multimodal_comprehensive.py -v --tb=short -m "not integration"',
            "description": "Comprehensive test coverage"
        },
        {
            "name": "All Multimodal Tests",
            "cmd": f'python -m pytest tests/test_multimodal.py tests/test_multimodal_edge_cases.py tests/test_multimodal_comprehensive.py --tb=line -m "not integration" -q',
            "description": "All multimodal unit tests combined"
        }
    ]
    
    results = []
    
    for suite in test_suites:
        success, output = run_command(suite["cmd"], suite["description"])
        results.append({
            "name": suite["name"],
            "success": success,
            "output": output
        })
    
    # Print summary
    print(f"\n\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}\n")
    
    passed_count = sum(1 for r in results if r["success"])
    total_count = len(results)
    
    for result in results:
        status = "[PASSED]" if result["success"] else "[FAILED]"
        print(f"{status} - {result['name']}")
    
    print(f"\n{'='*80}")
    print(f"Total: {passed_count}/{total_count} test suites passed")
    print(f"{'='*80}\n")
    
    # Return exit code
    sys.exit(0 if passed_count == total_count else 1)


if __name__ == "__main__":
    main()
