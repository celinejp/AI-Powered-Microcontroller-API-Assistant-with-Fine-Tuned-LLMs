#!/usr/bin/env python3
"""
QA System Demo - Microcontroller API Assistant
=============================================

This script demonstrates the complete QA system by running all test suites
and providing a comprehensive overview of the system's capabilities.
"""

import asyncio
import subprocess
import sys
import time
from pathlib import Path

def run_qa_demo():
    """Run a complete demonstration of the QA system."""
    print("🚀 MICROCONTROLLER API ASSISTANT - QA SYSTEM DEMO")
    print("=" * 60)
    print()
    
    # Check if virtual environment is activated
    if not Path("qa_venv").exists():
        print("❌ Virtual environment not found. Please run:")
        print("   python3 -m venv qa_venv")
        print("   source qa_venv/bin/activate")
        print("   pip install aiohttp click")
        return
    
    # Test suites to run
    test_suites = ["basic", "edge", "regression"]
    
    print("📋 Running Test Suites:")
    for suite in test_suites:
        print(f"   - {suite}")
    print()
    
    # Run each test suite
    results = {}
    for suite in test_suites:
        print(f"🔍 Running {suite.upper()} Test Suite...")
        print("-" * 40)
        
        try:
            # Run the test suite
            start_time = time.time()
            result = subprocess.run(
                ["python", "runner.py", "--test-suite", suite],
                capture_output=True,
                text=True,
                cwd=Path.cwd()
            )
            end_time = time.time()
            
            if result.returncode == 0:
                print("✅ Test suite completed successfully!")
                print(f"⏱️  Duration: {end_time - start_time:.2f}s")
                
                # Extract summary from output
                lines = result.stdout.split('\n')
                for line in lines:
                    if "Total Tests:" in line:
                        print(f"📊 {line.strip()}")
                    elif "Passed:" in line:
                        print(f"✅ {line.strip()}")
                    elif "Average Score:" in line:
                        print(f"🎯 {line.strip()}")
                        score_line = line
                        break
                
                # Parse score
                try:
                    score = float(score_line.split(":")[1].split("/")[0].strip())
                    results[suite] = {
                        "status": "PASS",
                        "score": score,
                        "duration": end_time - start_time
                    }
                except:
                    results[suite] = {
                        "status": "PASS",
                        "score": "N/A",
                        "duration": end_time - start_time
                    }
                
            else:
                print("❌ Test suite failed!")
                print(f"Error: {result.stderr}")
                results[suite] = {
                    "status": "FAIL",
                    "score": "N/A",
                    "duration": end_time - start_time
                }
                
        except Exception as e:
            print(f"❌ Error running {suite} test suite: {e}")
            results[suite] = {
                "status": "ERROR",
                "score": "N/A",
                "duration": 0
            }
        
        print()
    
    # Show summary
    print("📈 QA SYSTEM SUMMARY")
    print("=" * 60)
    
    total_tests = 0
    total_passed = 0
    total_score = 0
    total_duration = 0
    
    for suite, result in results.items():
        status_emoji = "✅" if result["status"] == "PASS" else "❌"
        print(f"{status_emoji} {suite.upper()}: {result['status']} | Score: {result['score']} | Duration: {result['duration']:.2f}s")
        
        if result["status"] == "PASS":
            total_passed += 1
            if isinstance(result["score"], (int, float)):
                total_score += result["score"]
        total_duration += result["duration"]
    
    print()
    print(f"🎯 Overall Results:")
    print(f"   - Test Suites: {len(test_suites)}")
    print(f"   - Passed: {total_passed}/{len(test_suites)}")
    print(f"   - Success Rate: {(total_passed/len(test_suites)*100):.1f}%")
    if total_score > 0:
        print(f"   - Average Score: {total_score/total_passed:.2f}/2.0")
    print(f"   - Total Duration: {total_duration:.2f}s")
    
    print()
    print("📁 Generated Reports:")
    report_dir = Path("qa_results")
    if report_dir.exists():
        for report_file in report_dir.glob("*_report.md"):
            print(f"   📄 {report_file.name}")
    
    print()
    print("🎉 QA SYSTEM DEMO COMPLETE!")
    print()
    print("💡 Key Features Demonstrated:")
    print("   ✅ Automated regression testing")
    print("   ✅ Intelligent scoring system (0-2 points)")
    print("   ✅ Keyword/function validation")
    print("   ✅ Edge case handling")
    print("   ✅ Professional Markdown reporting")
    print("   ✅ Modular architecture")
    print()
    print("🔧 Next Steps:")
    print("   1. Review generated reports in qa_results/")
    print("   2. Customize test cases in cases.py")
    print("   3. Adjust scoring weights in scoring.py")
    print("   4. Add new validation rules in validators.py")
    print("   5. Integrate into CI/CD pipeline")

if __name__ == "__main__":
    run_qa_demo()
