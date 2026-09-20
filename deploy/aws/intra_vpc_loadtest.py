#!/usr/bin/env python3
"""
Minimal load test, meant to run FROM an EC2 instance in the same VPC/AZ as
the app, hitting it over the private network - no public internet, no home
router in the path. Deliberately dependency-light (just aiohttp) so it's
fast to set up on a throwaway instance.

Usage: python3 intra_vpc_loadtest.py <app-private-ip> [concurrency ...]
"""
import asyncio
import sys
import time

import aiohttp


async def one(session, url):
    t0 = time.perf_counter()
    try:
        async with session.get(url) as r:
            await r.read()
            return (time.perf_counter() - t0) * 1000, r.status
    except Exception as e:
        return (time.perf_counter() - t0) * 1000, f"ERR:{e}"


async def run(concurrency, total, url):
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(concurrency)

        async def limited():
            async with sem:
                return await one(session, url)

        results = await asyncio.gather(*[limited() for _ in range(total)])
        times = sorted(r[0] for r in results)
        errors = [r for r in results if not isinstance(r[1], int) or r[1] >= 400]
        n = len(times)
        sub100 = sum(1 for t in times if t < 100) / n * 100
        print(
            f"concurrency={concurrency:>4} total={total} errors={len(errors)} "
            f"avg={sum(times)/n:7.1f}ms p50={times[n//2]:7.1f}ms "
            f"p95={times[int(n*0.95)]:7.1f}ms p99={times[int(n*0.99)]:7.1f}ms "
            f"sub100ms={sub100:5.1f}%"
        )


async def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    target = sys.argv[1]
    concurrencies = [int(a) for a in sys.argv[2:]] or [1, 10, 50, 100, 500]
    # Accept either a bare IP (assumed port 8000, single instance) or a full
    # http:// base URL (e.g. an ALB DNS name on port 80) - either way we hit /health.
    base = target if target.startswith("http") else f"http://{target}:8000"
    url = f"{base}/health"
    print(f"Hitting {url} (500 requests per concurrency level)")
    for c in concurrencies:
        await run(c, 500, url)


if __name__ == "__main__":
    asyncio.run(main())
