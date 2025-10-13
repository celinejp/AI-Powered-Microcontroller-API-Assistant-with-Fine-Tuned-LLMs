#!/usr/bin/env python3
"""
Automated QA System for Microcontroller API Assistant
=====================================================

This script provides comprehensive testing for the Microcontroller API Assistant,
including regression testing, scoring, keyword validation, and edge case handling.

Usage:
    python runner.py [--api-url URL] [--output-dir DIR] [--verbose]
"""

import asyncio
import json
import logging
import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp
import click
from dataclasses import dataclass, asdict

# Set UTF-8 encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

# CI Gatekeeper Thresholds
MIN_SUCCESS_RATE = 0.9  # 90% of test cases must pass
MIN_AVG_SCORE = 1.5     # Minimum average score on 0-2 scale

# Import test modules
from cases import TestCase, TestSuite
from scoring import CodeScorer
from validators import CodeValidator
from reporters import ReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qa_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Represents the result of a single test case."""
    test_case: TestCase
    input_prompt: str
    generated_code: str
    response_time: float
    latency_ms: float
    tokens_estimated: int
    tokens_per_sec: float
    score: float
    score_breakdown: Dict[str, float]
    validation_results: Dict[str, bool]
    warnings: List[str]
    errors: List[str]
    status: str  # 'pass', 'warn', 'fail'
    api_response: Dict = None


@dataclass
class TestSummary:
    """Summary of all test results."""
    total_tests: int
    passed_tests: int
    warned_tests: int
    failed_tests: int
    average_score: float
    total_time: float
    test_results: List[TestResult]
    accuracy_percentage: float = 0.0
    p50_latency: float = 0.0
    p95_latency: float = 0.0
    avg_tokens_per_sec: float = 0.0
    success_rate: float = 0.0
    avg_response_latency: float = 0.0


class QATestRunner:
    """Main QA test runner for the Microcontroller API Assistant."""
    
    def __init__(self, api_url: str = "http://localhost:8000", output_dir: str = "qa_results", 
                 temperature: float = 0.7, top_p: float = 0.9, seed: int = None):
        self.api_url = api_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create raw results directory
        self.raw_dir = self.output_dir / "raw"
        self.raw_dir.mkdir(exist_ok=True)
        
        # Generation parameters
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        
        # Initialize components
        self.scorer = CodeScorer()
        self.validator = CodeValidator()
        self.reporter = ReportGenerator()
        
        # Test results storage
        self.results: List[TestResult] = []
        
        # Performance tracking
        self.latencies: List[float] = []
        self.token_counts: List[int] = []
        
    async def run_test_case(self, session: aiohttp.ClientSession, test_case: TestCase) -> TestResult:
        """Run a single test case and return the result."""
        logger.info(f"Running test: {test_case.name}")
        
        start_time = time.time()
        
        try:
            # Prepare request payload with generation parameters
            payload = {
                "query": test_case.prompt,
                "temperature": self.temperature,
                "top_p": self.top_p
            }
            if self.seed is not None:
                payload["seed"] = self.seed
            
            # Make API request
            async with session.post(
                f"{self.api_url}/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response_time = time.time() - start_time
                latency_ms = response_time * 1000
                
                if response.status != 200:
                    return TestResult(
                        test_case=test_case,
                        input_prompt=test_case.prompt,
                        generated_code="",
                        response_time=response_time,
                        latency_ms=latency_ms,
                        tokens_estimated=0,
                        tokens_per_sec=0.0,
                        score=0.0,
                        score_breakdown={"api_error": 0.0},
                        validation_results={},
                        warnings=[f"API returned status {response.status}"],
                        errors=[f"HTTP {response.status} error"],
                        status="fail"
                    )
                
                data = await response.json()
                
                # Handle schema normalization
                if "text" in data and "response" not in data:
                    data["response"] = data["text"]
                    logger.warning(f"API returned 'text' instead of 'response' for test {test_case.name}")
                
                generated_code = data.get("response", "")
                
                # Extract performance metrics
                latency_ms = data.get("latency_ms", latency_ms)
                tokens_estimated = data.get("tokens_generated", 0)
                if tokens_estimated == 0:
                    from utils.text import count_tokens_estimate
                    tokens_estimated = count_tokens_estimate(generated_code)
                
                tokens_per_sec = tokens_estimated / (latency_ms / 1000) if latency_ms > 0 else 0.0
                
                # Track performance metrics
                self.latencies.append(latency_ms)
                self.token_counts.append(tokens_estimated)
                
        except Exception as e:
            response_time = time.time() - start_time
            latency_ms = response_time * 1000
            return TestResult(
                test_case=test_case,
                input_prompt=test_case.prompt,
                generated_code="",
                response_time=response_time,
                latency_ms=latency_ms,
                tokens_estimated=0,
                tokens_per_sec=0.0,
                score=0.0,
                score_breakdown={"exception": 0.0},
                validation_results={},
                warnings=[],
                errors=[f"Exception: {str(e)}"],
                status="fail"
            )
        
        # Score the generated code
        score_breakdown = self.scorer.score_code(
            generated_code, 
            test_case.expected_keywords,
            test_case.expected_libraries,
            test_case.microcontroller_type
        )
        total_score = score_breakdown["total"]  # Use normalized total score
        
        # Validate the code
        validation_results = self.validator.validate_code(
            generated_code,
            test_case.expected_keywords,
            test_case.expected_libraries,
            test_case.microcontroller_type,
            data  # Pass API response for schema validation
        )
        
        # Determine status
        warnings = []
        errors = []
        
        if total_score >= 1.5:
            status = "pass"
        elif total_score >= 0.5:
            status = "warn"
            warnings.append("Low score - code may have issues")
        else:
            status = "fail"
            errors.append("Very low score - significant issues detected")
        
        # Check for specific validation failures
        for check, passed in validation_results.items():
            if not passed:
                if check in ["syntax", "critical_keywords"]:
                    errors.append(f"Failed validation: {check}")
                else:
                    warnings.append(f"Failed validation: {check}")
        
        return TestResult(
            test_case=test_case,
            input_prompt=test_case.prompt,
            generated_code=generated_code,
            response_time=response_time,
            latency_ms=latency_ms,
            tokens_estimated=tokens_estimated,
            tokens_per_sec=tokens_per_sec,
            score=total_score,
            score_breakdown=score_breakdown,
            validation_results=validation_results,
            warnings=warnings,
            errors=errors,
            status=status,
            api_response=data
        )
    
    async def run_test_suite(self, test_suite: TestSuite) -> TestSummary:
        """Run a complete test suite."""
        logger.info(f"Starting test suite: {test_suite.name}")
        logger.info(f"API URL: {self.api_url}")
        
        start_time = time.time()
        
        # Create HTTP session
        async with aiohttp.ClientSession() as session:
            # Run all test cases
            tasks = [self.run_test_case(session, test_case) for test_case in test_suite.test_cases]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Test case {i} failed with exception: {result}")
                    results[i] = TestResult(
                        test_case=test_suite.test_cases[i],
                        input_prompt=test_suite.test_cases[i].prompt,
                        generated_code="",
                        response_time=0.0,
                        score=0.0,
                        score_breakdown={"exception": 0.0},
                        validation_results={},
                        warnings=[],
                        errors=[f"Exception: {str(result)}"],
                        status="fail"
                    )
        
        total_time = time.time() - start_time
        
        # Calculate summary with proper normalization
        passed = sum(1 for r in results if r.status == "pass")
        warned = sum(1 for r in results if r.status == "warn")
        failed = sum(1 for r in results if r.status == "fail")
        
        # Normalize average score to 0-2.0 range
        max_points_per_test = 2.0
        total_points = sum(r.score for r in results)
        avg_score = total_points / len(results) if results else 0.0
        accuracy_percentage = (avg_score / max_points_per_test) * 100
        
        summary = TestSummary(
            total_tests=len(results),
            passed_tests=passed,
            warned_tests=warned,
            failed_tests=failed,
            average_score=avg_score,
            total_time=total_time,
            test_results=results
        )
        
        # Add performance metrics to summary
        summary.accuracy_percentage = accuracy_percentage
        summary.p50_latency = self._calculate_percentile(self.latencies, 50) if self.latencies else 0
        summary.p95_latency = self._calculate_percentile(self.latencies, 95) if self.latencies else 0
        summary.avg_tokens_per_sec = sum(self.token_counts) / total_time if total_time > 0 else 0
        summary.success_rate = (passed / len(results)) if results else 0.0
        summary.avg_response_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0.0
        
        return summary
    
    def save_results(self, summary: TestSummary, test_suite_name: str):
        """Save test results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create suite-specific raw directory
        suite_raw_dir = self.raw_dir / test_suite_name / timestamp
        suite_raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results as JSON
        results_file = self.output_dir / f"{test_suite_name}_{timestamp}_results.json"
        
        # Convert to JSON-serializable format
        def convert_to_serializable(obj):
            if hasattr(obj, 'value'):  # Handle Enum types
                return obj.value
            return str(obj)
        
        summary_dict = asdict(summary)
        summary_dict['test_results'] = []
        
        for i, result in enumerate(summary.test_results):
            result_dict = asdict(result)
            result_dict['test_case'] = {
                'name': result.test_case.name,
                'prompt': result.test_case.prompt,
                'microcontroller_type': result.test_case.microcontroller_type.value,
                'expected_keywords': result.test_case.expected_keywords,
                'expected_libraries': result.test_case.expected_libraries,
                'description': result.test_case.description,
                'category': result.test_case.category,
                'difficulty': result.test_case.difficulty,
            }
            summary_dict['test_results'].append(result_dict)
            
            # Save individual test case raw response
            if result.api_response:
                case_file = suite_raw_dir / f"case_{i:02d}_{result.test_case.name}.json"
                with open(case_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'test_case': result_dict['test_case'],
                        'api_response': result.api_response,
                        'generated_code': result.generated_code,
                        'score': result.score,
                        'validation_results': result.validation_results
                    }, f, indent=2, ensure_ascii=False)
        
        # Add metadata
        summary_dict['metadata'] = {
            'timestamp': timestamp,
            'test_suite': test_suite_name,
            'generation_params': {
                'temperature': self.temperature,
                'top_p': self.top_p,
                'seed': self.seed
            },
            'git_info': self._get_git_info(),
            'runner_version': '1.0.0'
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(summary_dict, f, indent=2, ensure_ascii=False)
        
        # Generate and save report
        report_file = self.output_dir / f"{test_suite_name}_{timestamp}_report.md"
        report_content = self.reporter.generate_report(summary, test_suite_name, self.temperature, self.top_p, self.seed)
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Results saved to: {results_file}")
        logger.info(f"Report saved to: {report_file}")
        logger.info(f"Raw responses saved to: {suite_raw_dir}")
        
        return results_file, report_file
    
    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile from list of values."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = (percentile / 100) * (len(sorted_values) - 1)
        
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower = sorted_values[int(index)]
            upper = sorted_values[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def _get_git_info(self) -> Dict[str, str]:
        """Get git commit information if available."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                cwd=Path.cwd().parent  # Go to project root
            )
            commit_hash = result.stdout.strip() if result.returncode == 0 else "unknown"
            
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True,
                text=True,
                cwd=Path.cwd().parent
            )
            is_dirty = bool(result.stdout.strip()) if result.returncode == 0 else False
            
            return {
                "commit": commit_hash,
                "dirty": is_dirty
            }
        except Exception:
            return {
                "commit": "unknown",
                "dirty": False
            }


@click.command()
@click.option('--api-url', default='http://localhost:8000', help='API endpoint URL')
@click.option('--output-dir', default='qa_results', help='Output directory for results')
@click.option('--test-suite', default='all', help='Test suite to run (all, basic, edge, stress, regression)')
@click.option('--verbose', is_flag=True, help='Enable verbose logging')
@click.option('--temperature', default=0.7, type=float, help='Generation temperature')
@click.option('--top-p', default=0.9, type=float, help='Generation top-p')
@click.option('--seed', default=None, type=int, help='Random seed for reproducibility')
def main(api_url: str, output_dir: str, test_suite: str, verbose: bool, 
         temperature: float, top_p: float, seed: int):
    """Run QA tests for Microcontroller API Assistant."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting QA Test Runner")
    logger.info(f"API URL: {api_url}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Test Suite: {test_suite}")
    
    # Initialize test runner
    runner = QATestRunner(api_url, output_dir, temperature, top_p, seed)
    
    # Load test suites
    from cases import load_test_suites
    test_suites = load_test_suites()
    
    if test_suite == 'all':
        suites_to_run = test_suites
    elif test_suite in test_suites:
        suites_to_run = {test_suite: test_suites[test_suite]}
    else:
        logger.error(f"Unknown test suite: {test_suite}")
        logger.error(f"Available suites: {', '.join(test_suites.keys())}")
        sys.exit(1)
    
    # Run test suites
    suite_summaries = {}
    for suite_name, suite in suites_to_run.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Running Test Suite: {suite_name}")
        logger.info(f"{'='*50}")
        
        try:
            summary = asyncio.run(runner.run_test_suite(suite))
            suite_summaries[suite_name] = summary
            
            # Print summary
            logger.info(f"\nTest Suite Summary: {suite_name}")
            logger.info(f"Total Tests: {summary.total_tests}")
            logger.info(f"Passed: {summary.passed_tests} ✅")
            logger.info(f"Warned: {summary.warned_tests} ⚠️")
            logger.info(f"Failed: {summary.failed_tests} ❌")
            logger.info(f"Average Score: {summary.average_score:.2f}/2.0")
            logger.info(f"Accuracy: {summary.accuracy_percentage:.1f}%")
            logger.info(f"Success Rate: {summary.success_rate:.1%}")
            logger.info(f"P50 Latency: {summary.p50_latency:.1f}ms")
            logger.info(f"P95 Latency: {summary.p95_latency:.1f}ms")
            logger.info(f"Total Time: {summary.total_time:.2f}s")
            
            # Save results
            runner.save_results(summary, suite_name)
            
        except Exception as e:
            logger.error(f"Failed to run test suite {suite_name}: {e}")
            continue
    
    logger.info("\nQA Testing Complete!")
    
    # CI Gatekeeper Assessment
    logger.info("\n" + "="*60)
    logger.info("🔍 CI GATEKEEPER ASSESSMENT")
    logger.info("="*60)
    
    # Calculate overall metrics across all suites
    all_results = []
    all_latencies = []
    total_tests = 0
    total_passed = 0
    
    for suite_name, suite in suites_to_run.items():
        if suite_name in suite_summaries:
            summary = suite_summaries[suite_name]
            all_results.extend(summary.test_results)
            all_latencies.extend([r.latency_ms for r in summary.test_results])
            total_tests += summary.total_tests
            total_passed += summary.passed_tests
    
    # Calculate overall metrics
    overall_success_rate = total_passed / total_tests if total_tests > 0 else 0.0
    overall_avg_score = sum(r.score for r in all_results) / len(all_results) if all_results else 0.0
    overall_avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0.0
    
    # Check thresholds
    success_rate_ok = overall_success_rate >= MIN_SUCCESS_RATE
    avg_score_ok = overall_avg_score >= MIN_AVG_SCORE
    
    # Print CI Gatekeeper Summary
    logger.info(f"📊 OVERALL METRICS:")
    logger.info(f"   Total Tests: {total_tests}")
    logger.info(f"   Passed Tests: {total_passed}")
    logger.info(f"   Success Rate: {overall_success_rate:.1%} {'✅' if success_rate_ok else '❌'} (min: {MIN_SUCCESS_RATE:.1%})")
    logger.info(f"   Average Score: {overall_avg_score:.2f}/2.0 {'✅' if avg_score_ok else '❌'} (min: {MIN_AVG_SCORE:.1f})")
    logger.info(f"   Average Latency: {overall_avg_latency:.1f}ms")
    
    logger.info(f"\n🎯 THRESHOLD ASSESSMENT:")
    logger.info(f"   Success Rate: {success_rate_ok}")
    logger.info(f"   Average Score: {avg_score_ok}")
    
    # Final verdict
    all_thresholds_met = success_rate_ok and avg_score_ok
    
    if all_thresholds_met:
        logger.info(f"\n🎉 CI GATEKEEPER: PASSED ✅")
        logger.info(f"   All thresholds met! Ready for deployment.")
        return 0
    else:
        logger.info(f"\n🚨 CI GATEKEEPER: FAILED ❌")
        logger.info(f"   Thresholds not met. Please fix issues before deployment.")
        if not success_rate_ok:
            logger.info(f"   - Success rate {overall_success_rate:.1%} below minimum {MIN_SUCCESS_RATE:.1%}")
        if not avg_score_ok:
            logger.info(f"   - Average score {overall_avg_score:.2f} below minimum {MIN_AVG_SCORE:.1f}")
        return 1


if __name__ == "__main__":
    main()
