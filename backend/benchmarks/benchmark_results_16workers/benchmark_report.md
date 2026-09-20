# CareNavigator-AI Performance Benchmark Report

## Executive Summary

- **Total Requests**: 13,500
- **Successful Requests**: 12,000
- **Overall Success Rate**: 88.9%
- **Endpoints Tested**: 9
- **Max Concurrent Users**: 500

## Resume Claims Validation

### Overall Performance
- **Sub-100ms Response Rate**: 55.9%
- **Meets Sub-100ms Claim**: NO
- **Handles 1000+ Concurrent**: NO
- **Max Concurrent Tested**: 500

## Endpoint Performance Details

### health_check
- **Success Rate**: 100.0%
- **Average Response Time**: 124.34ms
- **P95 Response Time**: 325.93ms
- **P99 Response Time**: 405.50ms
- **Sub-100ms Rate**: 64.9%
- **Meets Performance Claims**: NO

### root
- **Success Rate**: 100.0%
- **Average Response Time**: 128.65ms
- **P95 Response Time**: 348.81ms
- **P99 Response Time**: 393.12ms
- **Sub-100ms Rate**: 65.0%
- **Meets Performance Claims**: NO

### metrics
- **Success Rate**: 100.0%
- **Average Response Time**: 159.21ms
- **P95 Response Time**: 413.45ms
- **P99 Response Time**: 445.54ms
- **Sub-100ms Rate**: 58.4%
- **Meets Performance Claims**: NO

### list_models
- **Success Rate**: 100.0%
- **Average Response Time**: 151.97ms
- **P95 Response Time**: 392.46ms
- **P99 Response Time**: 439.52ms
- **Sub-100ms Rate**: 62.5%
- **Meets Performance Claims**: NO

### update_registry
- **Success Rate**: 100.0%
- **Average Response Time**: 154.79ms
- **P95 Response Time**: 373.53ms
- **P99 Response Time**: 421.18ms
- **Sub-100ms Rate**: 39.9%
- **Meets Performance Claims**: NO

### insurance_match
- **Success Rate**: 100.0%
- **Average Response Time**: 191.26ms
- **P95 Response Time**: 478.43ms
- **P99 Response Time**: 513.63ms
- **Sub-100ms Rate**: 57.5%
- **Meets Performance Claims**: NO

### summarize_text
- **Success Rate**: 100.0%
- **Average Response Time**: 218.93ms
- **P95 Response Time**: 563.71ms
- **P99 Response Time**: 621.88ms
- **Sub-100ms Rate**: 49.5%
- **Meets Performance Claims**: NO

### reload_plans
- **Success Rate**: 100.0%
- **Average Response Time**: 183.36ms
- **P95 Response Time**: 472.90ms
- **P99 Response Time**: 505.16ms
- **Sub-100ms Rate**: 49.9%
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

