#!/usr/bin/env python3
"""
Multi-endpoint load test, meant to run FROM an EC2 instance in the same
VPC/AZ as the app, hitting it over the private network - no public
internet, no home router in the path.

Exercises every non-mutating, non-API-key endpoint EXCEPT /predict:
GET  /  /health  /status  /models  /metrics  /cache/stats  /performance
POST /insurance-match/  /summary

Usage: python3 intra_vpc_loadtest_endpoints.py <base-url> [concurrency ...]
  e.g. python3 intra_vpc_loadtest_endpoints.py http://some-alb-dns 1 10 50 100 500
"""
import asyncio
import sys
import time

import aiohttp

ENDPOINTS = [
    ("GET", "/", None),
    ("GET", "/health", None),
    ("GET", "/status", None),
    ("GET", "/models", None),
    ("GET", "/metrics", None),
    ("GET", "/cache/stats", None),
    ("GET", "/performance", None),
    ("POST", "/insurance-match/", {"description": "diabetes management plan with low premium"}),
    ("POST", "/summary", {"condition_name": "diabetes", "raw_text": "Diabetes is a chronic condition."}),
]


REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=15)


async def one(session, method, url, payload):
    t0 = time.perf_counter()
    try:
        if method == "GET":
            async with session.get(url, timeout=REQUEST_TIMEOUT) as r:
                await r.read()
                return (time.perf_counter() - t0) * 1000, r.status
        else:
            async with session.post(url, json=payload, timeout=REQUEST_TIMEOUT) as r:
                await r.read()
                return (time.perf_counter() - t0) * 1000, r.status
    except Exception as e:
        return (time.perf_counter() - t0) * 1000, f"ERR:{e}"


async def run_one_endpoint(session, sem, method, path, payload, base, concurrency, total):
    url = f"{base}{path}"

    async def limited():
        async with sem:
            return await one(session, method, url, payload)

    results = await asyncio.gather(*[limited() for _ in range(total)])
    times = sorted(r[0] for r in results)
    errors = [r for r in results if not isinstance(r[1], int) or r[1] >= 400]
    n = len(times)
    sub100 = sum(1 for t in times if t < 100) / n * 100
    print(
        f"  {method:4} {path:<16} concurrency={concurrency:>4} total={total} errors={len(errors):>3} "
        f"avg={sum(times)/n:7.1f}ms p50={times[n//2]:7.1f}ms "
        f"p95={times[int(n*0.95)]:7.1f}ms p99={times[int(n*0.99)]:7.1f}ms "
        f"sub100ms={sub100:5.1f}%",
        flush=True,
    )
    if errors[:3]:
        sample = [e[1] for e in errors[:3]]
        print(f"       sample errors: {sample}", flush=True)


async def warm_up(session, sem, base, concurrency):
    """Pre-establish `concurrency` keep-alive connections before measuring,
    so the first endpoint tested doesn't unfairly absorb connection-setup cost."""
    url = f"{base}/health"

    async def limited():
        async with sem:
            return await one(session, "GET", url, None)

    await asyncio.gather(*[limited() for _ in range(concurrency)])


async def run_concurrency_level(base, concurrency, total):
    print(f"--- concurrency={concurrency} ---", flush=True)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(concurrency)
        await warm_up(session, sem, base, concurrency)
        for method, path, payload in ENDPOINTS:
            await run_one_endpoint(session, sem, method, path, payload, base, concurrency, total)


async def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    target = sys.argv[1]
    concurrencies = [int(a) for a in sys.argv[2:]] or [1, 10, 50, 100, 500]
    base = target if target.startswith("http") else f"http://{target}:8000"
    print(f"Hitting {len(ENDPOINTS)} endpoints on {base} (500 requests per endpoint per concurrency level, /predict excluded)")
    for c in concurrencies:
        await run_concurrency_level(base, c, 500)


if __name__ == "__main__":
    asyncio.run(main())
