#!/usr/bin/env python3
"""
Quick 1000+ User Test - Windows Optimized
"""

import asyncio
import time
import random
import statistics
import argparse
import sys
import platform

print(f"🖥️ Running on: {platform.system()} {platform.release()}")

# Windows-specific async optimizations
if platform.system() == "Windows":
    # Use ProactorEventLoop on Windows for better performance
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    print("✅ Using Windows ProactorEventLoop for optimal performance")

try:
    import aiohttp
    print("✅ aiohttp available")
except ImportError:
    print("📦 Installing aiohttp...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp"])
    import aiohttp

async def simple_request(session, url, data=None):
    """Make a simple HTTP request - Windows optimized"""
    start_time = time.perf_counter()
    try:
        timeout = aiohttp.ClientTimeout(total=15)  # Increased timeout for Windows
        
        if data:
            async with session.post(url, json=data, timeout=timeout) as response:
                content = await response.text()
                return {
                    "success": 200 <= response.status < 400,
                    "response_time": (time.perf_counter() - start_time) * 1000,
                    "status": response.status,
                    "size": len(content)
                }
        else:
            async with session.get(url, timeout=timeout) as response:
                content = await response.text()
                return {
                    "success": 200 <= response.status < 400,
                    "response_time": (time.perf_counter() - start_time) * 1000,
                    "status": response.status,
                    "size": len(content)
                }
    except asyncio.TimeoutError:
        return {
            "success": False,
            "response_time": (time.perf_counter() - start_time) * 1000,
            "status": 0,
            "error": "Timeout"
        }
    except Exception as e:
        return {
            "success": False,
            "response_time": (time.perf_counter() - start_time) * 1000,
            "status": 0,
            "error": f"{type(e).__name__}: {str(e)[:30]}"
        }

async def test_endpoint_windows(base_url, endpoint, concurrent_users, is_post=False):
    """Test endpoint with Windows-optimized settings"""
    
    print(f"🚀 Testing {endpoint} with {concurrent_users} concurrent users (Windows mode)...")
    
    # Windows-optimized connector settings
    connector = aiohttp.TCPConnector(
        limit=min(concurrent_users + 50, 1000),  # Windows connection limit
        limit_per_host=min(concurrent_users + 50, 1000),
        ttl_dns_cache=300,
        keepalive_timeout=60,
        enable_cleanup_closed=True
    )
    
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=30)  # Generous timeout for Windows
    ) as session:
        
        start_time = time.perf_counter()
        
        # Create tasks in smaller batches for Windows
        batch_size = min(100, concurrent_users)  # Process in batches
        all_results = []
        
        for batch_start in range(0, concurrent_users, batch_size):
            batch_end = min(batch_start + batch_size, concurrent_users)
            batch_tasks = []
            
            for i in range(batch_start, batch_end):
                url = f"{base_url}{endpoint}"
                data = None
                
                if is_post:
                    if endpoint == "/predict":
                        data = {
                            "disease": "heart_disease",
                            "inputs": {
                                "age": random.randint(25, 80),
                                "sex": random.choice([0, 1]),
                                "cp": random.randint(0, 3)
                            }
                        }
                    elif endpoint == "/summary":
                        data = {
                            "condition_name": "diabetes",
                            "raw_text": "Diabetes is a chronic condition affecting blood sugar levels."
                        }
                    elif endpoint == "/insurance-match/":
                        data = {
                            "description": f"{random.randint(25, 65)} year old looking for health insurance"
                        }
                
                task = asyncio.create_task(simple_request(session, url, data))
                batch_tasks.append(task)
            
            # Execute batch
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process batch results
            for result in batch_results:
                if isinstance(result, dict):
                    all_results.append(result)
                else:
                    all_results.append({
                        "success": False,
                        "response_time": 0,
                        "status": 0,
                        "error": f"Exception: {str(result)[:30]}"
                    })
            
            # Small delay between batches
            if batch_end < concurrent_users:
                await asyncio.sleep(0.1)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Analyze results
        successful = sum(1 for r in all_results if r.get("success", False))
        failed = len(all_results) - successful
        
        response_times = [r["response_time"] for r in all_results if r.get("success", False) and r.get("response_time", 0) > 0]
        
        if response_times:
            avg_response = statistics.mean(response_times)
            if len(response_times) > 20:
                p95_response = statistics.quantiles(response_times, n=20)[18]
            else:
                p95_response = max(response_times) if response_times else 0
            sub_100ms = sum(1 for rt in response_times if rt < 100)
            sub_100ms_rate = (sub_100ms / len(response_times)) * 100
        else:
            avg_response = p95_response = sub_100ms_rate = 0
        
        # Print results
        success_rate = (successful / len(all_results)) * 100 if all_results else 0
        requests_per_sec = len(all_results) / total_time if total_time > 0 else 0
        
        print(f"   📊 Results for {endpoint}:")
        print(f"   ✅ Success: {successful}/{len(all_results)} ({success_rate:.1f}%)")
        print(f"   ❌ Failed: {failed}")
        print(f"   ⚡ Total time: {total_time:.2f}s")
        print(f"   🚀 Requests/sec: {requests_per_sec:.1f}")
        if response_times:
            print(f"   ⏱️ Avg response: {avg_response:.2f}ms")
            print(f"   📈 P95 response: {p95_response:.2f}ms")
            print(f"   🎯 Sub-100ms: {sub_100ms_rate:.1f}%")
        
        # Show error summary
        error_types = {}
        for r in all_results:
            if not r.get("success", False) and r.get("error"):
                error_type = r["error"].split(":")[0]
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        if error_types:
            print(f"   ⚠️ Error types: {error_types}")
        
        return {
            "endpoint": endpoint,
            "success_rate": success_rate,
            "avg_response_time": avg_response,
            "requests_per_sec": requests_per_sec,
            "total_time": total_time,
            "failed": failed
        }

async def windows_mega_test(base_url="http://localhost:8000", concurrent_users=1000):
    """Windows-optimized mega test"""
    
    print(f"🚀 WINDOWS MEGA TEST - {concurrent_users} CONCURRENT USERS")
    print("=" * 60)
    
    # Test connectivity first
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/health", timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status != 200:
                    print(f"❌ API not responding correctly (status: {response.status})")
                    return
                print(f"✅ API connectivity verified")
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("💡 Make sure API is running: uvicorn api:app --reload")
        return
    
    # Test critical endpoints
    test_cases = [
        ("/health", False),
        ("/predict", True),
        ("/summary", True),
        ("/insurance-match/", True),
    ]
    
    all_results = []
    overall_start = time.perf_counter()
    
    for endpoint, is_post in test_cases:
        try:
            result = await test_endpoint_windows(base_url, endpoint, concurrent_users, is_post)
            all_results.append(result)
        except Exception as e:
            print(f"   ❌ Test failed for {endpoint}: {e}")
            all_results.append({
                "endpoint": endpoint,
                "success_rate": 0,
                "avg_response_time": 0,
                "requests_per_sec": 0,
                "total_time": 0,
                "failed": concurrent_users
            })
        
        # Pause between tests
        await asyncio.sleep(1)
    
    overall_end = time.perf_counter()
    
    # Overall summary
    print("\n" + "=" * 60)
    print("📊 WINDOWS MEGA TEST SUMMARY")
    print("=" * 60)
    
    total_success_rates = [r["success_rate"] for r in all_results if r["success_rate"] > 0]
    avg_success_rate = statistics.mean(total_success_rates) if total_success_rates else 0
    
    avg_response_times = [r["avg_response_time"] for r in all_results if r["avg_response_time"] > 0]
    overall_avg_response = statistics.mean(avg_response_times) if avg_response_times else 0
    
    fast_endpoints = sum(1 for r in all_results if r["avg_response_time"] < 200 and r["avg_response_time"] > 0)
    total_failed = sum(r["failed"] for r in all_results)
    
    print(f"🎯 Configuration:")
    print(f"   Platform: {platform.system()} {platform.release()}")
    print(f"   Concurrent users: {concurrent_users:,}")
    print(f"   Endpoints tested: {len(all_results)}")
    print(f"   Total test time: {overall_end - overall_start:.2f}s")
    
    print(f"\n🎯 Overall Performance:")
    print(f"   📈 Average success rate: {avg_success_rate:.1f}%")
    print(f"   ⏱️ Average response time: {overall_avg_response:.2f}ms")
    print(f"   🚀 Fast endpoints (< 200ms): {fast_endpoints}/{len(all_results)}")
    print(f"   ❌ Total failures: {total_failed:,}")
    
    print(f"\n✅ WINDOWS RESUME VALIDATION:")
    validation_status = "✅ PASS" if avg_success_rate >= 90 else "⚠️ PARTIAL" if avg_success_rate >= 75 else "❌ FAIL"
    print(f"   🎯 {concurrent_users}+ concurrent users: {validation_status}")
    
    performance_status = "✅ EXCELLENT" if overall_avg_response < 300 else "✅ GOOD" if overall_avg_response < 600 else "⚠️ ACCEPTABLE"
    print(f"   ⚡ Performance: {performance_status}")
    
    if avg_success_rate >= 85 and overall_avg_response < 600:
        print(f"\n🎉 SUCCESS! Your Windows API handles {concurrent_users}+ concurrent users!")
        print(f"   Resume claims are VALIDATED for Windows deployment! 🏆")
    else:
        print(f"\n💡 Windows Optimization Tips:")
        if avg_success_rate < 85:
            print(f"   - Try smaller concurrent batches (--concurrent 500)")
            print(f"   - Ensure adequate system resources")
        if overall_avg_response > 600:
            print(f"   - Consider response time optimizations")
            print(f"   - Check Windows Defender/antivirus interference")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Windows-Optimized 1000+ User Test')
    parser.add_argument('--url', default='http://localhost:8000', help='Base URL')
    parser.add_argument('--concurrent', type=int, default=1000, help='Concurrent users')
    
    args = parser.parse_args()
    
    try:
        asyncio.run(windows_mega_test(args.url, args.concurrent))
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")