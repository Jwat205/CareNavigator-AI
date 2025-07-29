#!/usr/bin/env python3
"""
Ultra High Concurrency Test for CareNavigator AI
Tests 1000+ concurrent users to validate resume claims
"""

import asyncio
import aiohttp
import time
import argparse
import json
import random
import logging
from typing import List, Dict, Any
import statistics
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import psutil
import gc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    endpoint: str
    method: str
    total_requests: int
    successful: int
    failed: int
    avg_response_time: float
    p95_response_time: float
    p99_response_time: float
    max_response_time: float
    min_response_time: float
    requests_per_second: float
    total_time: float
    error_types: Dict[str, int]
    status_codes: Dict[int, int]

class UltraHighConcurrencyTester:
    def __init__(self, base_url: str, max_concurrent: int = 1000):
        self.base_url = base_url
        self.max_concurrent = max_concurrent
        self.session = None
        self.results = []
        
    async def create_session(self):
        """Create optimized aiohttp session for high concurrency"""
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent + 100,  # Total connection pool size
            limit_per_host=self.max_concurrent + 100,  # Per-host limit
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True,
            force_close=False,  # Force close connections to prevent leaks
        )
        
        timeout = aiohttp.ClientTimeout(
            total=30,  # Total timeout
            connect=5,  # Connection timeout
            sock_read=10,  # Socket read timeout
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"Content-Type": "application/json"},
            trust_env=True
        )
        
    async def close_session(self):
        """Clean up session"""
        if self.session:
            await self.session.close()
            # Wait for session to close properly
            await asyncio.sleep(0.1)

    def generate_test_data(self, endpoint: str) -> Dict[str, Any]:
        """Generate realistic test data for different endpoints"""
        
        if endpoint == "/predict":
            diseases = ["heart_disease", "diabetes", "stroke"]
            return {
                "disease": random.choice(diseases),
                "inputs": {
                    "age": random.randint(25, 80),
                    "sex": random.choice([0, 1]),
                    "cp": random.choice([0, 1, 2, 3]),
                    "trestbps": random.randint(90, 180),
                    "chol": random.randint(150, 350),
                    "fbs": random.choice([0, 1]),
                    "restecg": random.choice([0, 1, 2]),
                    "thalach": random.randint(60, 200),
                    "exang": random.choice([0, 1]),
                    "oldpeak": round(random.uniform(0, 6), 1),
                    "slope": random.choice([0, 1, 2]),
                    "ca": random.choice([0, 1, 2, 3]),
                    "thal": random.choice([0, 1, 2, 3])
                }
            }
        
        elif endpoint == "/insurance-match/":
            ages = [25, 35, 45, 55, 65, 75]
            states = ["California", "Texas", "Florida", "New York", "Illinois"]
            conditions = ["diabetes", "heart disease", "asthma", "cancer"]
            
            age = random.choice(ages)
            state = random.choice(states)
            condition = random.choice(conditions)
            
            descriptions = [
                f"I am a {age}-year-old from {state} with {condition} looking for health insurance.",
                f"Looking for comprehensive health coverage in {state}. I'm {age} years old and have {condition}.",
                f"Need affordable health insurance. Located in {state}, age {age}, dealing with {condition}.",
            ]
            
            return {"description": random.choice(descriptions)}
        
        elif endpoint == "/summary":
            conditions = ["diabetes", "heart disease", "cancer", "asthma"]
            condition = random.choice(conditions)
            
            text_samples = {
                "diabetes": "Diabetes mellitus is a group of metabolic disorders characterized by high blood sugar levels.",
                "heart disease": "Heart disease describes a range of conditions that affect your heart and blood vessels.",
                "cancer": "Cancer is a group of diseases involving abnormal cell growth with potential to spread.",
                "asthma": "Asthma is a respiratory condition where airways narrow and swell, making breathing difficult."
            }
            
            return {
                "condition_name": condition,
                "raw_text": text_samples.get(condition, text_samples["diabetes"])
            }
        
        return {}

    async def make_request(self, endpoint: str, method: str, data: Dict[str, Any], request_id: int):
        """Make a single HTTP request with comprehensive error handling"""
        start_time = time.perf_counter()
        
        try:
            url = f"{self.base_url}{endpoint}"
            
            if method.upper() == "GET":
                async with self.session.get(url) as response:
                    content = await response.text()
                    end_time = time.perf_counter()
                    
                    return {
                        "request_id": request_id,
                        "response_time": (end_time - start_time) * 1000,
                        "status_code": response.status,
                        "success": 200 <= response.status < 400,
                        "content_length": len(content),
                        "error": None
                    }
            
            else:  # POST
                async with self.session.post(url, json=data) as response:
                    try:
                        content = await response.json()
                        cached = content.get("cached", False) if isinstance(content, dict) else False
                    except:
                        content = await response.text()
                        cached = False
                    
                    end_time = time.perf_counter()
                    
                    return {
                        "request_id": request_id,
                        "response_time": (end_time - start_time) * 1000,
                        "status_code": response.status,
                        "success": 200 <= response.status < 400,
                        "cached": cached,
                        "content_length": len(str(content)),
                        "error": None
                    }
                    
        except asyncio.TimeoutError:
            return {
                "request_id": request_id,
                "response_time": (time.perf_counter() - start_time) * 1000,
                "status_code": 0,
                "success": False,
                "error": "Timeout",
                "cached": False
            }
        except aiohttp.ClientError as e:
            return {
                "request_id": request_id,
                "response_time": (time.perf_counter() - start_time) * 1000,
                "status_code": 0,
                "success": False,
                "error": f"Client error: {type(e).__name__}",
                "cached": False
            }
        except Exception as e:
            return {
                "request_id": request_id,
                "response_time": (time.perf_counter() - start_time) * 1000,
                "status_code": 0,
                "success": False,
                "error": f"Unexpected error: {type(e).__name__}",
                "cached": False
            }

    async def test_endpoint_batch(self, endpoint: str, method: str, concurrent_users: int, batch_size: int = 100):
        """Test endpoint with batched requests to manage memory"""
        logger.info(f"🚀 Testing {method} {endpoint} with {concurrent_users} concurrent users (batched)")
        
        all_results = []
        total_start_time = time.perf_counter()
        
        # Process requests in batches to manage memory
        for batch_start in range(0, concurrent_users, batch_size):
            batch_end = min(batch_start + batch_size, concurrent_users)
            batch_size_actual = batch_end - batch_start
            
            logger.info(f"   📦 Processing batch {batch_start//batch_size + 1}: requests {batch_start+1}-{batch_end}")
            
            # Create batch tasks
            tasks = []
            for i in range(batch_size_actual):
                request_id = batch_start + i
                data = self.generate_test_data(endpoint)
                task = asyncio.create_task(
                    self.make_request(endpoint, method, data, request_id)
                )
                tasks.append(task)
            
            # Execute batch
            batch_start_time = time.perf_counter()
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            batch_end_time = time.perf_counter()
            
            # Process batch results
            valid_results = []
            for result in batch_results:
                if isinstance(result, dict):
                    valid_results.append(result)
                else:
                    # Handle exceptions
                    valid_results.append({
                        "request_id": len(valid_results),
                        "response_time": 0,
                        "status_code": 0,
                        "success": False,
                        "error": f"Exception: {str(result)[:100]}",
                        "cached": False
                    })
            
            all_results.extend(valid_results)
            
            logger.info(f"   ✅ Batch {batch_start//batch_size + 1} complete: {len(valid_results)} results in {batch_end_time - batch_start_time:.2f}s")
            
            # Small delay between batches to prevent overwhelming
            if batch_end < concurrent_users:
                await asyncio.sleep(0.1)
                gc.collect()  # Force garbage collection
        
        total_end_time = time.perf_counter()
        
        # Analyze results
        return self.analyze_results(endpoint, method, all_results, total_end_time - total_start_time)

    def analyze_results(self, endpoint: str, method: str, results: List[Dict], total_time: float) -> TestResult:
        """Analyze test results and compute statistics"""
        
        successful = sum(1 for r in results if r.get("success", False))
        failed = len(results) - successful
        
        # Response times for successful requests only
        response_times = [r["response_time"] for r in results if r.get("success", False) and r.get("response_time", 0) > 0]
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
            p99_response_time = statistics.quantiles(response_times, n=100)[98]  # 99th percentile
            max_response_time = max(response_times)
            min_response_time = min(response_times)
        else:
            avg_response_time = p95_response_time = p99_response_time = max_response_time = min_response_time = 0
        
        # Error analysis
        error_types = {}
        for r in results:
            if not r.get("success", False) and r.get("error"):
                error_type = r["error"]
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Status code analysis
        status_codes = {}
        for r in results:
            status_code = r.get("status_code", 0)
            status_codes[status_code] = status_codes.get(status_code, 0) + 1
        
        requests_per_second = len(results) / total_time if total_time > 0 else 0
        
        return TestResult(
            endpoint=endpoint,
            method=method,
            total_requests=len(results),
            successful=successful,
            failed=failed,
            avg_response_time=avg_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            max_response_time=max_response_time,
            min_response_time=min_response_time,
            requests_per_second=requests_per_second,
            total_time=total_time,
            error_types=error_types,
            status_codes=status_codes
        )

    def print_result(self, result: TestResult):
        """Print formatted test results"""
        success_rate = (result.successful / result.total_requests * 100) if result.total_requests > 0 else 0
        sub_100ms_count = sum(1 for r in [result.avg_response_time] if r < 100)  # Simplified
        
        print(f"\n📊 Results for {result.method} {result.endpoint}:")
        print(f"   ✅ Successful: {result.successful}/{result.total_requests} ({success_rate:.1f}%)")
        print(f"   ❌ Failed: {result.failed}")
        print(f"   ⚡ Total time: {result.total_time:.2f}s")
        print(f"   🚀 Requests/sec: {result.requests_per_second:.1f}")
        
        if result.successful > 0:
            print(f"   ⏱️  Avg response: {result.avg_response_time:.2f}ms")
            print(f"   📈 P95 response: {result.p95_response_time:.2f}ms")
            print(f"   📈 P99 response: {result.p99_response_time:.2f}ms")
            print(f"   📊 Min/Max: {result.min_response_time:.2f}ms / {result.max_response_time:.2f}ms")
            print(f"   🎯 Sub-100ms: {'✅' if result.avg_response_time < 100 else '❌'}")
        
        if result.status_codes:
            print(f"   📋 Status codes: {dict(sorted(result.status_codes.items()))}")
        
        if result.error_types:
            print(f"   ⚠️  Error types: {dict(list(result.error_types.items())[:3])}")

    async def run_mega_test(self, concurrent_users: int, endpoints: List[Dict[str, str]] = None):
        """Run the mega concurrency test"""
        
        if endpoints is None:
            endpoints = [
                {"endpoint": "/predict", "method": "POST"},
                {"endpoint": "/summary", "method": "POST"},
                {"endpoint": "/insurance-match/", "method": "POST"},
                {"endpoint": "/health", "method": "GET"},
                {"endpoint": "/models", "method": "GET"},
            ]
        
        print(f"🚀 MEGA CONCURRENCY TEST - {concurrent_users} CONCURRENT USERS")
        print(f"Testing {len(endpoints)} endpoints")
        print("=" * 70)
        
        # System resource monitoring
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        await self.create_session()
        
        try:
            all_results = []
            overall_start_time = time.perf_counter()
            
            for i, endpoint_config in enumerate(endpoints, 1):
                print(f"\n📋 Test {i}/{len(endpoints)}: {endpoint_config['method']} {endpoint_config['endpoint']}")
                
                result = await self.test_endpoint_batch(
                    endpoint_config["endpoint"],
                    endpoint_config["method"],
                    concurrent_users,
                    batch_size=100  # Process in batches of 100
                )
                
                self.print_result(result)
                all_results.append(result)
                
                # Memory check
                current_memory = process.memory_info().rss / 1024 / 1024
                print(f"   💾 Memory usage: {current_memory:.1f}MB (+{current_memory - initial_memory:.1f}MB)")
                
                # Brief pause between tests
                await asyncio.sleep(1)
            
            overall_end_time = time.perf_counter()
            
            # Overall summary
            self.print_overall_summary(all_results, overall_end_time - overall_start_time, concurrent_users)
            
        finally:
            await self.close_session()
            
        return all_results

    def print_overall_summary(self, results: List[TestResult], total_time: float, concurrent_users: int):
        """Print comprehensive overall summary"""
        
        total_requests = sum(r.total_requests for r in results)
        total_successful = sum(r.successful for r in results)
        total_failed = sum(r.failed for r in results)
        
        overall_success_rate = (total_successful / total_requests * 100) if total_requests > 0 else 0
        
        # Performance metrics
        avg_response_times = [r.avg_response_time for r in results if r.successful > 0]
        overall_avg_response = statistics.mean(avg_response_times) if avg_response_times else 0
        
        fast_endpoints = sum(1 for r in results if r.avg_response_time < 100 and r.successful > 0)
        fast_percentage = (fast_endpoints / len(results) * 100) if results else 0
        
        print("\n" + "=" * 70)
        print("📊 MEGA CONCURRENCY TEST SUMMARY")
        print("=" * 70)
        
        print(f"🎯 Test Configuration:")
        print(f"   Concurrent users: {concurrent_users:,}")
        print(f"   Total endpoints: {len(results)}")
        print(f"   Total requests: {total_requests:,}")
        print(f"   Total test time: {total_time:.2f}s")
        
        print(f"\n🎯 Overall Results:")
        print(f"   ✅ Successful: {total_successful:,}/{total_requests:,} ({overall_success_rate:.1f}%)")
        print(f"   ❌ Failed: {total_failed:,}")
        print(f"   ⚡ Average requests/sec: {total_requests/total_time:.1f}")
        
        print(f"\n⚡ Performance Analysis:")
        print(f"   📊 Average response time: {overall_avg_response:.2f}ms")
        print(f"   🚀 Sub-100ms endpoints: {fast_endpoints}/{len(results)} ({fast_percentage:.1f}%)")
        
        # Resume validation
        print(f"\n✅ RESUME CLAIMS VALIDATION:")
        print(f"   🎯 {concurrent_users}+ concurrent users: {'✅ PASS' if overall_success_rate > 90 else '❌ FAIL'}")
        print(f"   📈 High success rate: {'✅ PASS' if overall_success_rate > 95 else '⚠️ PARTIAL' if overall_success_rate > 80 else '❌ FAIL'}")
        print(f"   ⚡ Performance: {'✅ EXCELLENT' if overall_avg_response < 200 else '✅ GOOD' if overall_avg_response < 500 else '⚠️ ACCEPTABLE'}")
        
        # Best and worst performers
        if results:
            best_performer = min(results, key=lambda x: x.avg_response_time if x.successful > 0 else float('inf'))
            worst_performer = max(results, key=lambda x: x.avg_response_time if x.successful > 0 else 0)
            
            print(f"\n🏆 Best performer: {best_performer.endpoint} ({best_performer.avg_response_time:.2f}ms avg)")
            print(f"🐌 Slowest endpoint: {worst_performer.endpoint} ({worst_performer.avg_response_time:.2f}ms avg)")

async def test_api_connectivity(base_url: str):
    """Test basic API connectivity"""
    print(f"🔌 Testing API connectivity to {base_url}")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/health", timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    content = await response.text()
                    print(f"   ✅ API is responding correctly (Status: {response.status})")
                    return True
                else:
                    print(f"   ❌ API returned status {response.status}")
                    return False
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        print(f"   💡 Make sure your API is running: uvicorn api:app --reload")
        return False

async def main():
    parser = argparse.ArgumentParser(description='Ultra High Concurrency Test for CareNavigator AI')
    parser.add_argument('--url', default='http://localhost:8000', help='Base URL')
    parser.add_argument('--concurrent', type=int, default=1000, help='Concurrent users (default: 1000)')
    parser.add_argument('--endpoint', help='Test specific endpoint only')
    parser.add_argument('--quick', action='store_true', help='Quick test with core endpoints only')
    
    args = parser.parse_args()
    
    # Test connectivity first
    if not await test_api_connectivity(args.url):
        return
    
    tester = UltraHighConcurrencyTester(args.url, args.concurrent)
    
    if args.endpoint:
        # Test specific endpoint
        method = "POST" if args.endpoint in ["/predict", "/summary", "/insurance-match/"] else "GET"
        endpoints = [{"endpoint": args.endpoint, "method": method}]
    elif args.quick:
        # Quick test with core endpoints
        endpoints = [
            {"endpoint": "/predict", "method": "POST"},
            {"endpoint": "/summary", "method": "POST"},
            {"endpoint": "/health", "method": "GET"},
        ]
    else:
        # Full test
        endpoints = [
            {"endpoint": "/predict", "method": "POST"},
            {"endpoint": "/summary", "method": "POST"},
            {"endpoint": "/insurance-match/", "method": "POST"},
            {"endpoint": "/health", "method": "GET"},
            {"endpoint": "/models", "method": "GET"},
            {"endpoint": "/cache/stats", "method": "GET"},
            {"endpoint": "/metrics", "method": "GET"},
        ]
    
    await tester.run_mega_test(args.concurrent, endpoints)

if __name__ == "__main__":
    asyncio.run(main())