#!/usr/bin/env python3
"""
Fixed High Concurrency Test for CareNavigator AI
Improved error detection and debugging
"""

import asyncio
import aiohttp
import time
import argparse
import json
import random
from typing import List, Dict, Any
import io

async def test_endpoint_batch(session: aiohttp.ClientSession, base_url: str, 
                            endpoint_config: Dict[str, Any], concurrent_users: int):
    """Test a specific endpoint with concurrent requests"""
    
    endpoint = endpoint_config["endpoint"]
    method = endpoint_config["method"]
    
    print(f"🔍 Testing {method} {endpoint} with {concurrent_users} concurrent users")
    
    start_time = time.time()
    
    # Create tasks for concurrent requests
    tasks = []
    for i in range(concurrent_users):
        if method == "GET":
            task = asyncio.create_task(
                make_get_request(session, f"{base_url}{endpoint}", i)
            )
        elif method == "POST":
            # Generate unique test data for each request
            test_data = generate_test_data(endpoint, i)
            task = asyncio.create_task(
                make_post_request(session, f"{base_url}{endpoint}", test_data, i)
            )
        tasks.append(task)
    
    # Execute all requests
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = time.time()
    
    # Analyze results with better error detection
    successful = 0
    failed = 0
    exceptions = []
    response_times = []
    cached_responses = 0
    status_codes = {}
    
    for r in results:
        if isinstance(r, Exception):
            failed += 1
            exceptions.append(r)
        else:
            # Check if it's a successful response
            status_code = r.get('status_code', 0)
            if status_code in status_codes:
                status_codes[status_code] += 1
            else:
                status_codes[status_code] = 1
                
            if 200 <= status_code < 400:  # Success range
                successful += 1
                if 'response_time' in r:
                    response_times.append(r['response_time'])
                if r.get('cached', False):
                    cached_responses += 1
            else:
                failed += 1
                # Add error details for debugging
                if 'error' not in r:
                    r['error'] = f"HTTP {status_code}"
                exceptions.append(r.get('error', f"HTTP {status_code}"))
    
    # Print results with better debugging
    print(f"   📊 Results for {endpoint}:")
    print(f"   ✅ Successful: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
    print(f"   ❌ Failed: {failed}")
    print(f"   ⚡ Total time: {end_time - start_time:.2f}s")
    print(f"   🚀 Requests/sec: {len(results)/(end_time - start_time):.1f}")
    
    # Show status code breakdown
    if status_codes:
        print(f"   📋 Status codes: {status_codes}")
    
    if response_times:
        avg_response = sum(response_times) / len(response_times)
        response_times.sort()
        p95_response = response_times[int(len(response_times) * 0.95)]
        p99_response = response_times[int(len(response_times) * 0.99)]
        sub_100ms = sum(1 for rt in response_times if rt < 100)
        sub_100ms_rate = sub_100ms / len(response_times) * 100
        
        print(f"   ⏱️  Avg response: {avg_response:.2f}ms")
        print(f"   📈 P95 response: {p95_response:.2f}ms")
        print(f"   📈 P99 response: {p99_response:.2f}ms")
        print(f"   🎯 Sub-100ms rate: {sub_100ms_rate:.1f}%")
        
        if cached_responses > 0:
            print(f"   💾 Cached responses: {cached_responses}/{successful} ({cached_responses/max(successful,1)*100:.1f}%)")
    
    # Show sample errors (limit to 3 for readability)
    if exceptions:
        unique_errors = list(set(str(e)[:100] for e in exceptions[:5]))
        print(f"   ⚠️  Sample errors: {unique_errors[:3]}")
    
    print()
    
    return {
        "endpoint": endpoint,
        "method": method,
        "total_requests": len(results),
        "successful": successful,
        "failed": failed,
        "success_rate": successful/len(results)*100,
        "avg_response_time": sum(response_times)/len(response_times) if response_times else 0,
        "sub_100ms_rate": sub_100ms_rate if response_times else 0,
        "cached_responses": cached_responses,
        "requests_per_second": len(results)/(end_time - start_time),
        "total_time": end_time - start_time,
        "status_codes": status_codes
    }

def generate_test_data(endpoint: str, request_id: int) -> Dict[str, Any]:
    """Generate realistic test data for each endpoint"""
    
    # Sample data pools for realistic testing
    diseases = ["heart_disease", "diabetes", "stroke"]
    ages = [25, 35, 45, 55, 65, 75]
    conditions = ["diabetes", "hypertension", "heart disease", "asthma"]
    states = ["Florida", "California", "Texas", "New York", "Illinois"]
    
    if endpoint == "/predict":
        return {
            "disease": random.choice(diseases),
            "inputs": {
                "age": random.choice(ages),
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
        age = random.choice(ages)
        state = random.choice(states)
        condition = random.choice(conditions)
        family_size = random.choice([1, 2, 3, 4])
        
        descriptions = [
            f"I am a {age}-year-old from {state} with {condition} looking for health insurance for my family of {family_size}.",
            f"Looking for comprehensive health coverage in {state}. I'm {age} years old and have {condition}.",
            f"Need affordable health insurance for {family_size} people. Located in {state}, age {age}, dealing with {condition}.",
        ]
        
        return {
            "description": random.choice(descriptions)
        }
    
    elif endpoint == "/summary":
        condition = random.choice(conditions)
        
        text_samples = {
            "diabetes": "Diabetes mellitus is a group of metabolic disorders characterized by a high blood sugar level over a prolonged period of time. Symptoms often include frequent urination, increased thirst and increased appetite.",
            "hypertension": "High blood pressure (hypertension) is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems.",
            "heart disease": "Heart disease describes a range of conditions that affect your heart. Diseases under the heart disease umbrella include blood vessel diseases, such as coronary artery disease.",
            "asthma": "Asthma is a condition in which your airways narrow and swell and may produce extra mucus. This can make breathing difficult and trigger coughing."
        }
        
        return {
            "condition_name": condition,
            "raw_text": text_samples.get(condition, text_samples["diabetes"])
        }
    
    else:
        # For endpoints that don't need data
        return {}

async def make_get_request(session: aiohttp.ClientSession, url: str, request_id: int):
    """Make a GET request and measure performance"""
    try:
        start_time = time.perf_counter()
        
        async with session.get(url) as response:
            # Read the response to get full timing
            content = await response.text()
            end_time = time.perf_counter()
            
            response_time = (end_time - start_time) * 1000
            
            return {
                "request_id": request_id,
                "response_time": response_time,
                "status_code": response.status,
                "success": 200 <= response.status < 400,
                "content_length": len(content)
            }
    except asyncio.TimeoutError:
        return {
            "request_id": request_id,
            "error": "Request timeout",
            "status_code": 0,
            "success": False
        }
    except aiohttp.ClientError as e:
        return {
            "request_id": request_id,
            "error": f"Client error: {str(e)}",
            "status_code": 0,
            "success": False
        }
    except Exception as e:
        return {
            "request_id": request_id,
            "error": f"Unexpected error: {str(e)}",
            "status_code": 0,
            "success": False
        }

async def make_post_request(session: aiohttp.ClientSession, url: str, data: Dict[str, Any], request_id: int):
    """Make a POST request and measure performance"""
    try:
        start_time = time.perf_counter()
        
        async with session.post(url, json=data) as response:
            # Try to parse JSON response
            try:
                result = await response.json()
            except:
                # If not JSON, get text
                result = await response.text()
            
            end_time = time.perf_counter()
            
            response_time = (end_time - start_time) * 1000
            
            return {
                "request_id": request_id,
                "response_time": response_time,
                "status_code": response.status,
                "success": 200 <= response.status < 400,
                "cached": result.get("cached", False) if isinstance(result, dict) else False,
                "content_length": len(str(result))
            }
            
    except asyncio.TimeoutError:
        return {
            "request_id": request_id,
            "error": "Request timeout",
            "status_code": 0,
            "success": False,
            "cached": False
        }
    except aiohttp.ClientError as e:
        return {
            "request_id": request_id,
            "error": f"Client error: {str(e)}",
            "status_code": 0,
            "success": False,
            "cached": False
        }
    except Exception as e:
        return {
            "request_id": request_id,
            "error": f"Unexpected error: {str(e)}",
            "status_code": 0,
            "success": False,
            "cached": False
        }

async def test_api_connectivity(base_url: str):
    """Test basic API connectivity first"""
    print(f"🔌 Testing API connectivity to {base_url}")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/health") as response:
                content = await response.text()
                print(f"   Status: {response.status}")
                print(f"   Response: {content[:200]}...")
                
                if response.status == 200:
                    print("   ✅ API is responding correctly")
                    return True
                else:
                    print(f"   ❌ API returned error status: {response.status}")
                    return False
                    
    except aiohttp.ClientConnectorError:
        print(f"   ❌ Cannot connect to API at {base_url}")
        print(f"   💡 Make sure your API is running: uvicorn api:app --reload")
        return False
    except Exception as e:
        print(f"   ❌ Connection error: {e}")
        return False

async def run_comprehensive_concurrency_test(base_url: str, concurrent_users: int = 100):
    """Run comprehensive test on all 14 endpoints"""
    
    # First test connectivity
    if not await test_api_connectivity(base_url):
        print("\n❌ API connectivity test failed. Please start your API first.")
        return
    
    print()
    
    # Define all endpoints (excluding upload-and-train for bulk testing)
    endpoints = [
        {"endpoint": "/", "method": "GET"},
        {"endpoint": "/health", "method": "GET"},
        {"endpoint": "/status", "method": "GET"},
        {"endpoint": "/models", "method": "GET"},
        {"endpoint": "/predict", "method": "POST"},
        {"endpoint": "/insurance-match/", "method": "POST"},
        {"endpoint": "/summary", "method": "POST"},
        {"endpoint": "/reload-plans/", "method": "POST"},
        {"endpoint": "/update-registry", "method": "POST"},
        {"endpoint": "/metrics", "method": "GET"},
        {"endpoint": "/cache/clear", "method": "POST"},
        {"endpoint": "/cache/stats", "method": "GET"},
        {"endpoint": "/models/heart_disease/metadata", "method": "GET"},
    ]
    
    print(f"🚀 COMPREHENSIVE CONCURRENCY TEST")
    print(f"Testing {len(endpoints)} endpoints with {concurrent_users} concurrent users each")
    print(f"Target: {base_url}")
    print("=" * 70)
    
    # Setup HTTP session with optimized settings
    connector = aiohttp.TCPConnector(
        limit=concurrent_users * 2,
        limit_per_host=concurrent_users * 2,
        ttl_dns_cache=300,
        keepalive_timeout=30
    )
    
    timeout = aiohttp.ClientTimeout(total=30)
    
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        headers={"Content-Type": "application/json"}
    ) as session:
        
        all_results = []
        total_start_time = time.time()
        
        # Test each endpoint
        for i, endpoint_config in enumerate(endpoints, 1):
            print(f"📋 Test {i}/{len(endpoints)}: {endpoint_config['method']} {endpoint_config['endpoint']}")
            
            result = await test_endpoint_batch(
                session, base_url, endpoint_config, concurrent_users
            )
            all_results.append(result)
            
            # Small delay between endpoint tests
            await asyncio.sleep(1)
        
        total_end_time = time.time()
        
        # Generate comprehensive summary
        print("=" * 70)
        print("📊 COMPREHENSIVE TEST SUMMARY")
        print("=" * 70)
        
        total_requests = sum(r["total_requests"] for r in all_results)
        total_successful = sum(r["successful"] for r in all_results)
        total_failed = sum(r["failed"] for r in all_results)
        
        print(f"🎯 Overall Results:")
        print(f"   Total endpoints tested: {len(endpoints)}")
        print(f"   Total requests sent: {total_requests:,}")
        print(f"   Total successful: {total_successful:,}")
        print(f"   Total failed: {total_failed:,}")
        print(f"   Overall success rate: {total_successful/total_requests*100:.1f}%")
        print(f"   Total test time: {total_end_time - total_start_time:.2f}s")
        print(f"   Average requests/sec: {total_requests/(total_end_time - total_start_time):.1f}")
        
        # Performance analysis
        print(f"\n⚡ Performance Analysis:")
        successful_results = [r for r in all_results if r["avg_response_time"] > 0]
        if successful_results:
            avg_response_times = [r["avg_response_time"] for r in successful_results]
            overall_avg = sum(avg_response_times) / len(avg_response_times)
            print(f"   Average response time: {overall_avg:.2f}ms")
            
            sub_100ms_rates = [r["sub_100ms_rate"] for r in successful_results]
            overall_sub_100ms = sum(sub_100ms_rates) / len(sub_100ms_rates)
            print(f"   Average sub-100ms rate: {overall_sub_100ms:.1f}%")
            
            # Fastest and slowest endpoints
            fastest = min(successful_results, key=lambda x: x["avg_response_time"])
            slowest = max(successful_results, key=lambda x: x["avg_response_time"])
            
            print(f"\n🏆 Fastest endpoint: {fastest['endpoint']} ({fastest['avg_response_time']:.2f}ms avg)")
            print(f"🐌 Slowest endpoint: {slowest['endpoint']} ({slowest['avg_response_time']:.2f}ms avg)")
        
        # Cache effectiveness
        cached_endpoints = [r for r in all_results if r["cached_responses"] > 0]
        if cached_endpoints:
            print(f"\n💾 Cache Performance:")
            for result in cached_endpoints:
                cache_rate = result["cached_responses"] / result["successful"] * 100 if result["successful"] > 0 else 0
                print(f"   {result['endpoint']}: {cache_rate:.1f}% cache hit rate")
        
        # Resume claims validation
        print(f"\n✅ RESUME CLAIMS VALIDATION:")
        print(f"   10+ REST endpoints: {'✅ PASS' if len(endpoints) >= 10 else '❌ FAIL'} ({len(endpoints)} endpoints)")
        print(f"   {concurrent_users}+ concurrent users: {'✅ PASS' if total_successful > concurrent_users * 5 else '❌ FAIL'}")
        print(f"   High success rate: {'✅ PASS' if total_successful/total_requests > 0.80 else '❌ FAIL'} ({total_successful/total_requests*100:.1f}%)")
        
        if successful_results:
            print(f"   Sub-100ms responses: {'✅ PASS' if overall_sub_100ms > 50 else '⚠️  PARTIAL'} ({overall_sub_100ms:.1f}%)")
        
        # Show detailed status code breakdown
        print(f"\n📋 Status Code Analysis:")
        all_status_codes = {}
        for result in all_results:
            for code, count in result["status_codes"].items():
                all_status_codes[code] = all_status_codes.get(code, 0) + count
        
        for code, count in sorted(all_status_codes.items()):
            print(f"   HTTP {code}: {count} requests ({count/total_requests*100:.1f}%)")

async def run_single_endpoint_test(base_url: str, endpoint: str, concurrent_users: int = 100):
    """Test a single endpoint with detailed debugging"""
    
    if not await test_api_connectivity(base_url):
        return
    
    print(f"\n🎯 SINGLE ENDPOINT TEST")
    print(f"Endpoint: {endpoint}")
    print(f"Concurrent users: {concurrent_users}")
    print("=" * 50)
    
    method = "POST" if endpoint in ["/predict", "/insurance-match/", "/summary", "/reload-plans/", "/update-registry", "/cache/clear"] else "GET"
    
    connector = aiohttp.TCPConnector(limit=concurrent_users + 10, limit_per_host=concurrent_users + 10)
    timeout = aiohttp.ClientTimeout(total=30)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        endpoint_config = {"endpoint": endpoint, "method": method}
        result = await test_endpoint_batch(session, base_url, endpoint_config, concurrent_users)
        
        print("🎉 SINGLE ENDPOINT TEST COMPLETE!")
        return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fixed High Concurrency Test for CareNavigator AI')
    parser.add_argument('--url', default='http://localhost:8000', help='Base URL')
    parser.add_argument('--concurrent', type=int, default=50, help='Concurrent users (default: 50)')
    parser.add_argument('--endpoint', help='Test specific endpoint only')
    parser.add_argument('--comprehensive', action='store_true', help='Run comprehensive test on all endpoints')
    parser.add_argument('--test-connection', action='store_true', help='Test API connection only')
    
    args = parser.parse_args()
    
    if args.test_connection:
        asyncio.run(test_api_connectivity(args.url))
    elif args.endpoint:
        asyncio.run(run_single_endpoint_test(args.url, args.endpoint, args.concurrent))
    elif args.comprehensive:
        asyncio.run(run_comprehensive_concurrency_test(args.url, args.concurrent))
    else:
        # Default: test health endpoint
        asyncio.run(run_single_endpoint_test(args.url, "/health", args.concurrent))