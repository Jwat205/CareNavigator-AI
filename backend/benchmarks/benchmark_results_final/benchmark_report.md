# CareNavigator-AI Performance Benchmark Report

## Executive Summary

- **Total Requests**: 22,500
- **Successful Requests**: 20,000
- **Overall Success Rate**: 88.9%
- **Endpoints Tested**: 9
- **Max Concurrent Users**: 500

## Resume Claims Validation

### Overall Performance
- **Sub-100ms Response Rate**: 72.7%
- **Meets Sub-100ms Claim**: NO
- **Handles 1000+ Concurrent**: NO
- **Max Concurrent Tested**: 500

## Endpoint Performance Details

### health_check
- **Success Rate**: 100.0%
- **Average Response Time**: 104.09ms
- **P95 Response Time**: 502.96ms
- **P99 Response Time**: 554.65ms
- **Sub-100ms Rate**: 74.2%
- **Meets Performance Claims**: NO

### root
- **Success Rate**: 100.0%
- **Average Response Time**: 89.40ms
- **P95 Response Time**: 377.36ms
- **P99 Response Time**: 397.21ms
- **Sub-100ms Rate**: 74.2%
- **Meets Performance Claims**: NO

### metrics
- **Success Rate**: 100.0%
- **Average Response Time**: 77.94ms
- **P95 Response Time**: 314.15ms
- **P99 Response Time**: 344.24ms
- **Sub-100ms Rate**: 73.8%
- **Meets Performance Claims**: NO

### list_models
- **Success Rate**: 100.0%
- **Average Response Time**: 61.74ms
- **P95 Response Time**: 221.22ms
- **P99 Response Time**: 249.13ms
- **Sub-100ms Rate**: 75.6%
- **Meets Performance Claims**: NO

### update_registry
- **Success Rate**: 100.0%
- **Average Response Time**: 86.23ms
- **P95 Response Time**: 306.59ms
- **P99 Response Time**: 331.34ms
- **Sub-100ms Rate**: 69.5%
- **Meets Performance Claims**: NO

### insurance_match
- **Success Rate**: 100.0%
- **Average Response Time**: 81.67ms
- **P95 Response Time**: 257.06ms
- **P99 Response Time**: 284.89ms
- **Sub-100ms Rate**: 73.3%
- **Meets Performance Claims**: NO

### summarize_text
- **Success Rate**: 100.0%
- **Average Response Time**: 112.42ms
- **P95 Response Time**: 400.66ms
- **P99 Response Time**: 416.87ms
- **Sub-100ms Rate**: 67.6%
- **Meets Performance Claims**: NO

### reload_plans
- **Success Rate**: 100.0%
- **Average Response Time**: 74.21ms
- **P95 Response Time**: 269.57ms
- **P99 Response Time**: 282.68ms
- **Sub-100ms Rate**: 73.5%
- **Meets Performance Claims**: NO

## Recommendations

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

