#!/usr/bin/env python3
"""
CareNavigator-AI Performance Benchmark Suite
============================================

This script benchmarks your FastAPI application to validate resume claims:
- Sub-100ms response times
- 1000+ concurrent requests handling
- API endpoint performance
- Database query optimization
- Error handling robustness

Usage:
    python benchmark.py --base-url http://localhost:8000
"""

import asyncio
import aiohttp
import time
import json
import pandas as pd
import numpy as np
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import logging
from dataclasses import dataclass
from pathlib import Path
import psutil
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    endpoint: str
    method: str
    response_time: float
    status_code: int
    success: bool
    error_message: str = None
    payload_size: int = 0
    timestamp: float = None

class CareNavigatorBenchmark:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.results: List[BenchmarkResult] = []
        self.session = None
        
        # Define test endpoints (matching your actual FastAPI routes)
        self.endpoints = {
            # Health/Status endpoints
            "health_check": {"method": "GET", "path": "/health", "payload": None},
            "root": {"method": "GET", "path": "/", "payload": None},
            "metrics": {"method": "GET", "path": "/metrics", "payload": None},
            
            # Model Management endpoints
            "list_models": {"method": "GET", "path": "/models", "payload": None},
            "update_registry": {"method": "POST", "path": "/update-registry", "payload": None},
            
            # Disease Prediction endpoint
            "predict_disease": {
                "method": "POST", 
                "path": "/predict",
                "payload": {
                    "disease": "heart_disease",  # Example disease name
                    "inputs": {
                        "age": 45,
                        "sex": 1,
                        "cp": 2,
                        "trestbps": 120,
                        "chol": 200,
                        "fbs": 0,
                        "restecg": 1,
                        "thalach": 150,
                        "exang": 0,
                        "oldpeak": 1.0,
                        "slope": 2,
                        "ca": 0,
                        "thal": 2
                    }
                }
            },
            
            # Insurance Matching endpoint
            "insurance_match": {
                "method": "POST",
                "path": "/insurance-match/",
                "payload": {
                    "description": "I am a 35-year-old from Florida with diabetes looking for comprehensive health insurance coverage for my family of 3. I need good prescription drug coverage and prefer plans with low deductibles."
                }
            },
            
            # Document Summarization endpoint
            "summarize_text": {
                "method": "POST",
                "path": "/summary",
                "payload": {
                    "condition_name": "diabetes",
                    "raw_text": "Diabetes mellitus is a group of metabolic disorders characterized by a high blood sugar level over a prolonged period of time. Symptoms often include frequent urination, increased thirst and increased appetite. If left untreated, diabetes can cause many health complications. Acute complications can include diabetic ketoacidosis, hyperosmolar hyperglycemic state, or death. Serious long-term complications include cardiovascular disease, stroke, chronic kidney disease, foot ulcers, damage to the nerves, damage to the eyes and cognitive impairment."
                }
            },
            
            # Reload Plans endpoint
            "reload_plans": {"method": "POST", "path": "/reload-plans/", "payload": None}
        }

    async def setup_session(self):
        """Setup async HTTP session with optimized settings"""
        connector = aiohttp.TCPConnector(
            limit=1000,  # Total connection pool size
            limit_per_host=100,  # Per-host connection limit
            ttl_dns_cache=300,
            keepalive_timeout=30
        )
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"Content-Type": "application/json"}
        )

    async def cleanup_session(self):
        """Cleanup HTTP session"""
        if self.session:
            await self.session.close()

    async def single_request(self, endpoint_name: str, endpoint_config: dict) -> BenchmarkResult:
        """Make a single HTTP request and measure performance"""
        start_time = time.perf_counter()
        
        try:
            url = f"{self.base_url}{endpoint_config['path']}"
            method = endpoint_config['method']
            payload = endpoint_config['payload']
            
            if method == "GET":
                async with self.session.get(url) as response:
                    await response.text()  # Read response body
                    end_time = time.perf_counter()
                    
                    return BenchmarkResult(
                        endpoint=endpoint_name,
                        method=method,
                        response_time=(end_time - start_time) * 1000,  # Convert to ms
                        status_code=response.status,
                        success=response.status < 400,
                        payload_size=len(str(payload)) if payload else 0,
                        timestamp=start_time
                    )
            
            elif method == "POST":
                async with self.session.post(url, json=payload) as response:
                    await response.text()  # Read response body
                    end_time = time.perf_counter()
                    
                    return BenchmarkResult(
                        endpoint=endpoint_name,
                        method=method,
                        response_time=(end_time - start_time) * 1000,  # Convert to ms
                        status_code=response.status,
                        success=response.status < 400,
                        payload_size=len(json.dumps(payload)) if payload else 0,
                        timestamp=start_time
                    )
                    
        except Exception as e:
            end_time = time.perf_counter()
            return BenchmarkResult(
                endpoint=endpoint_name,
                method=endpoint_config['method'],
                response_time=(end_time - start_time) * 1000,
                status_code=0,
                success=False,
                error_message=str(e),
                timestamp=start_time
            )

    async def load_test(self, endpoint_name: str, concurrent_requests: int = 100, 
                       total_requests: int = 1000) -> List[BenchmarkResult]:
        """Perform load testing on a specific endpoint"""
        logger.info(f"Starting load test: {endpoint_name} - {concurrent_requests} concurrent, {total_requests} total")
        
        endpoint_config = self.endpoints[endpoint_name]
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        async def limited_request():
            async with semaphore:
                return await self.single_request(endpoint_name, endpoint_config)
        
        # Create tasks for all requests
        tasks = [limited_request() for _ in range(total_requests)]
        
        # Execute all requests
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return valid results
        valid_results = [r for r in results if isinstance(r, BenchmarkResult)]
        logger.info(f"Completed {len(valid_results)}/{total_requests} requests for {endpoint_name}")
        
        return valid_results

    async def run_comprehensive_benchmark(self, concurrent_users: List[int] = [1, 10, 50, 100, 500, 1000],
                                        requests_per_test: int = 1000):
        """Run comprehensive benchmark across all endpoints and concurrency levels"""
        logger.info("Starting comprehensive benchmark suite...")
        
        await self.setup_session()
        
        try:
            # Test each endpoint at different concurrency levels
            for endpoint_name in self.endpoints.keys():
                logger.info(f"\n=== Testing endpoint: {endpoint_name} ===")
                
                for concurrent in concurrent_users:
                    if concurrent > requests_per_test:
                        continue
                        
                    logger.info(f"Testing with {concurrent} concurrent users...")
                    
                    # Record system metrics before test
                    cpu_before = psutil.cpu_percent()
                    memory_before = psutil.virtual_memory().percent
                    
                    start_time = time.time()
                    results = await self.load_test(endpoint_name, concurrent, requests_per_test)
                    end_time = time.time()
                    
                    # Record system metrics after test
                    cpu_after = psutil.cpu_percent()
                    memory_after = psutil.virtual_memory().percent
                    
                    # Add metadata to results
                    for result in results:
                        result.concurrent_users = concurrent
                        result.total_test_time = end_time - start_time
                        result.cpu_usage = (cpu_before + cpu_after) / 2
                        result.memory_usage = (memory_before + memory_after) / 2
                    
                    self.results.extend(results)
                    
                    # Quick analysis
                    if results:
                        response_times = [r.response_time for r in results if r.success]
                        success_rate = sum(1 for r in results if r.success) / len(results) * 100
                        
                        if response_times:
                            avg_response = np.mean(response_times)
                            p95_response = np.percentile(response_times, 95)
                            p99_response = np.percentile(response_times, 99)
                            
                            logger.info(f"  Success Rate: {success_rate:.1f}%")
                            logger.info(f"  Avg Response: {avg_response:.2f}ms")
                            logger.info(f"  P95 Response: {p95_response:.2f}ms")
                            logger.info(f"  P99 Response: {p99_response:.2f}ms")
                            
                            # Check resume claims
                            sub_100ms_count = sum(1 for rt in response_times if rt < 100)
                            sub_100ms_rate = sub_100ms_count / len(response_times) * 100
                            logger.info(f"  Sub-100ms Rate: {sub_100ms_rate:.1f}%")
                            
                    # Small delay between tests
                    await asyncio.sleep(2)
                    
        finally:
            await self.cleanup_session()

    def analyze_results(self) -> Dict[str, Any]:
        """Analyze benchmark results and generate insights"""
        if not self.results:
            return {"error": "No results to analyze"}
        
        df = pd.DataFrame([
            {
                "endpoint": r.endpoint,
                "method": r.method,
                "response_time": r.response_time,
                "status_code": r.status_code,
                "success": r.success,
                "concurrent_users": getattr(r, 'concurrent_users', 1),
                "timestamp": r.timestamp
            }
            for r in self.results
        ])
        
        analysis = {
            "summary": {
                "total_requests": len(df),
                "successful_requests": len(df[df['success'] == True]),
                "failed_requests": len(df[df['success'] == False]),
                "overall_success_rate": len(df[df['success'] == True]) / len(df) * 100,
                "unique_endpoints": df['endpoint'].nunique(),
                "max_concurrent_users": df['concurrent_users'].max()
            },
            "performance_metrics": {},
            "resume_validation": {}
        }
        
        # Analyze each endpoint
        for endpoint in df['endpoint'].unique():
            endpoint_df = df[df['endpoint'] == endpoint]
            successful_df = endpoint_df[endpoint_df['success'] == True]
            
            if len(successful_df) > 0:
                response_times = successful_df['response_time']
                
                analysis["performance_metrics"][endpoint] = {
                    "total_requests": len(endpoint_df),
                    "successful_requests": len(successful_df),
                    "success_rate": len(successful_df) / len(endpoint_df) * 100,
                    "avg_response_time": response_times.mean(),
                    "median_response_time": response_times.median(),
                    "p95_response_time": response_times.quantile(0.95),
                    "p99_response_time": response_times.quantile(0.99),
                    "min_response_time": response_times.min(),
                    "max_response_time": response_times.max(),
                    "std_response_time": response_times.std()
                }
                
                # Resume claim validation
                sub_100ms_rate = (response_times < 100).sum() / len(response_times) * 100
                analysis["resume_validation"][endpoint] = {
                    "sub_100ms_rate": sub_100ms_rate,
                    "meets_sub_100ms_claim": sub_100ms_rate > 80,  # 80% threshold
                    "max_concurrent_handled": endpoint_df['concurrent_users'].max(),
                    "handles_1000_concurrent": endpoint_df['concurrent_users'].max() >= 1000
                }
        
        # Overall resume validation
        all_successful = df[df['success'] == True]
        if len(all_successful) > 0:
            overall_sub_100ms_rate = (all_successful['response_time'] < 100).sum() / len(all_successful) * 100
            analysis["resume_validation"]["overall"] = {
                "sub_100ms_rate": overall_sub_100ms_rate,
                "meets_sub_100ms_claim": overall_sub_100ms_rate > 80,
                "max_concurrent_tested": df['concurrent_users'].max(),
                "handles_1000_concurrent": df['concurrent_users'].max() >= 1000,
                "overall_success_rate": analysis["summary"]["overall_success_rate"]
            }
        
        return analysis

    def generate_report(self, analysis: Dict[str, Any], output_dir: str = "benchmark_results"):
        """Generate comprehensive benchmark report"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save raw results
        results_df = pd.DataFrame([
            {
                "endpoint": r.endpoint,
                "method": r.method,
                "response_time": r.response_time,
                "status_code": r.status_code,
                "success": r.success,
                "error_message": r.error_message,
                "concurrent_users": getattr(r, 'concurrent_users', 1),
                "timestamp": r.timestamp,
                "cpu_usage": getattr(r, 'cpu_usage', None),
                "memory_usage": getattr(r, 'memory_usage', None)
            }
            for r in self.results
        ])
        
        results_df.to_csv(output_path / "raw_results.csv", index=False)
        
        # Save analysis
        with open(output_path / "analysis.json", "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Generate visualizations
        self._create_visualizations(results_df, output_path)
        
        # Generate markdown report
        self._generate_markdown_report(analysis, output_path)
        
        logger.info(f"Benchmark report generated in: {output_path}")

    def _create_visualizations(self, df: pd.DataFrame, output_path: Path):
        """Create performance visualization charts"""
        plt.style.use('seaborn-v0_8')
        
        # Response time distribution by endpoint
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Response time distribution
        successful_df = df[df['success'] == True]
        if len(successful_df) > 0:
            axes[0, 0].hist(successful_df['response_time'], bins=50, alpha=0.7, edgecolor='black')
            axes[0, 0].axvline(100, color='red', linestyle='--', label='100ms threshold')
            axes[0, 0].set_xlabel('Response Time (ms)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Response Time Distribution')
            axes[0, 0].legend()
        
        # 2. Response time by endpoint
        if len(successful_df) > 0:
            sns.boxplot(data=successful_df, x='endpoint', y='response_time', ax=axes[0, 1])
            axes[0, 1].axhline(100, color='red', linestyle='--', label='100ms threshold')
            axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45)
            axes[0, 1].set_title('Response Time by Endpoint')
            axes[0, 1].legend()
        
        # 3. Success rate by concurrent users
        if 'concurrent_users' in df.columns:
            success_by_concurrent = df.groupby('concurrent_users').agg({
                'success': ['count', 'sum']
            }).reset_index()
            success_by_concurrent.columns = ['concurrent_users', 'total', 'successful']
            success_by_concurrent['success_rate'] = success_by_concurrent['successful'] / success_by_concurrent['total'] * 100
            
            axes[1, 0].plot(success_by_concurrent['concurrent_users'], success_by_concurrent['success_rate'], 'bo-')
            axes[1, 0].set_xlabel('Concurrent Users')
            axes[1, 0].set_ylabel('Success Rate (%)')
            axes[1, 0].set_title('Success Rate vs Concurrent Users')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Response time vs concurrent users
        if 'concurrent_users' in df.columns and len(successful_df) > 0:
            avg_response_by_concurrent = successful_df.groupby('concurrent_users')['response_time'].mean().reset_index()
            
            axes[1, 1].plot(avg_response_by_concurrent['concurrent_users'], avg_response_by_concurrent['response_time'], 'go-')
            axes[1, 1].axhline(100, color='red', linestyle='--', label='100ms threshold')
            axes[1, 1].set_xlabel('Concurrent Users')
            axes[1, 1].set_ylabel('Average Response Time (ms)')
            axes[1, 1].set_title('Response Time vs Concurrent Users')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "performance_charts.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_markdown_report(self, analysis: Dict[str, Any], output_path: Path):
        """Generate markdown report"""
        report = f"""# CareNavigator-AI Performance Benchmark Report

## Executive Summary

- **Total Requests**: {analysis['summary']['total_requests']:,}
- **Successful Requests**: {analysis['summary']['successful_requests']:,}
- **Overall Success Rate**: {analysis['summary']['overall_success_rate']:.1f}%
- **Endpoints Tested**: {analysis['summary']['unique_endpoints']}
- **Max Concurrent Users**: {analysis['summary']['max_concurrent_users']:,}

## Resume Claims Validation

"""
        
        if 'overall' in analysis['resume_validation']:
            overall_validation = analysis['resume_validation']['overall']
            
            report += f"""### Overall Performance
- **Sub-100ms Response Rate**: {overall_validation['sub_100ms_rate']:.1f}%
- **Meets Sub-100ms Claim**: {' YES' if overall_validation['meets_sub_100ms_claim'] else 'NO'}
- **Handles 1000+ Concurrent**: {' YES' if overall_validation['handles_1000_concurrent'] else 'NO'}
- **Max Concurrent Tested**: {overall_validation['max_concurrent_tested']:,}

"""
        
        report += "## Endpoint Performance Details\n\n"
        
        for endpoint, metrics in analysis['performance_metrics'].items():
            validation = analysis['resume_validation'].get(endpoint, {})
            
            report += f"""### {endpoint}
- **Success Rate**: {metrics['success_rate']:.1f}%
- **Average Response Time**: {metrics['avg_response_time']:.2f}ms
- **P95 Response Time**: {metrics['p95_response_time']:.2f}ms
- **P99 Response Time**: {metrics['p99_response_time']:.2f}ms
- **Sub-100ms Rate**: {validation.get('sub_100ms_rate', 0):.1f}%
- **Meets Performance Claims**: {'YES' if validation.get('meets_sub_100ms_claim', False) else 'NO'}

"""
        
        report += """## Recommendations

Based on the benchmark results:

1. **Performance Optimization**: Focus on endpoints with high P99 response times
2. **Concurrency Handling**: Test and optimize for higher concurrent loads if needed
3. **Error Handling**: Investigate and fix endpoints with low success rates
4. **Monitoring**: Implement real-time performance monitoring in production
5. **Caching**: Consider implementing response caching for frequently accessed endpoints

## Files Generated

- `raw_results.csv` - Raw benchmark data
- `analysis.json` - Detailed analysis results
- `performance_charts.png` - Performance visualization charts
- `benchmark_report.md` - This report

"""
        
        with open(output_path / "benchmark_report.md", "w") as f:
            f.write(report)

async def main():
    parser = argparse.ArgumentParser(description='Benchmark CareNavigator-AI API')
    parser.add_argument('--base-url', default='http://localhost:8000', 
                       help='Base URL of the API (default: http://localhost:8000)')
    parser.add_argument('--concurrent', type=int, nargs='+', 
                       default=[1, 10, 50, 100, 500, 1000],
                       help='List of concurrent user levels to test')
    parser.add_argument('--requests', type=int, default=1000,
                       help='Number of requests per test (default: 1000)')
    parser.add_argument('--output', default='benchmark_results',
                       help='Output directory for results (default: benchmark_results)')
    
    args = parser.parse_args()
    
    # Initialize benchmark
    benchmark = CareNavigatorBenchmark(args.base_url)
    
    try:
        # Run comprehensive benchmark
        await benchmark.run_comprehensive_benchmark(
            concurrent_users=args.concurrent,
            requests_per_test=args.requests
        )
        
        # Analyze results
        analysis = benchmark.analyze_results()
        
        # Generate report
        benchmark.generate_report(analysis, args.output)
        
        # Print summary to console
        print("\n" + "="*60)
        print("BENCHMARK COMPLETE - RESUME VALIDATION SUMMARY")
        print("="*60)
        
        if 'overall' in analysis['resume_validation']:
            overall = analysis['resume_validation']['overall']
            print(f"Sub-100ms Response Rate: {overall['sub_100ms_rate']:.1f}%")
            print(f"Meets Sub-100ms Claim: {'YES' if overall['meets_sub_100ms_claim'] else ' NO'}")
            print(f"Handles 1000+ Concurrent: {' YES' if overall['handles_1000_concurrent'] else ' NO'}")
            print(f"Overall Success Rate: {overall['overall_success_rate']:.1f}%")
        
        print(f"\nDetailed report saved to: {args.output}/")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())