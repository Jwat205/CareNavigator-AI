# CareNavigator-AI Performance Benchmark Report

## Executive Summary

- **Total Requests**: 22,500
- **Successful Requests**: 20,000
- **Overall Success Rate**: 88.9%
- **Endpoints Tested**: 9
- **Max Concurrent Users**: 500

## Resume Claims Validation

### Overall Performance
- **Sub-100ms Response Rate**: 72.8%
- **Meets Sub-100ms Claim**: NO
- **Handles 1000+ Concurrent**: NO
- **Max Concurrent Tested**: 500

## Endpoint Performance Details

### health_check
- **Success Rate**: 100.0%
- **Average Response Time**: 71.82ms
- **P95 Response Time**: 202.07ms
- **P99 Response Time**: 259.89ms
- **Sub-100ms Rate**: 85.9%
- **Meets Performance Claims**: YES

### root
- **Success Rate**: 100.0%
- **Average Response Time**: 83.20ms
- **P95 Response Time**: 221.50ms
- **P99 Response Time**: 285.13ms
- **Sub-100ms Rate**: 77.9%
- **Meets Performance Claims**: NO

### metrics
- **Success Rate**: 100.0%
- **Average Response Time**: 88.87ms
- **P95 Response Time**: 254.64ms
- **P99 Response Time**: 317.05ms
- **Sub-100ms Rate**: 76.2%
- **Meets Performance Claims**: NO

### list_models
- **Success Rate**: 100.0%
- **Average Response Time**: 91.79ms
- **P95 Response Time**: 283.03ms
- **P99 Response Time**: 401.61ms
- **Sub-100ms Rate**: 79.0%
- **Meets Performance Claims**: NO

### update_registry
- **Success Rate**: 100.0%
- **Average Response Time**: 94.11ms
- **P95 Response Time**: 310.05ms
- **P99 Response Time**: 382.85ms
- **Sub-100ms Rate**: 78.5%
- **Meets Performance Claims**: NO

### insurance_match
- **Success Rate**: 100.0%
- **Average Response Time**: 113.71ms
- **P95 Response Time**: 323.85ms
- **P99 Response Time**: 389.13ms
- **Sub-100ms Rate**: 55.7%
- **Meets Performance Claims**: NO

### summarize_text
- **Success Rate**: 100.0%
- **Average Response Time**: 114.84ms
- **P95 Response Time**: 325.42ms
- **P99 Response Time**: 378.16ms
- **Sub-100ms Rate**: 55.2%
- **Meets Performance Claims**: NO

### reload_plans
- **Success Rate**: 100.0%
- **Average Response Time**: 97.23ms
- **P95 Response Time**: 308.94ms
- **P99 Response Time**: 361.37ms
- **Sub-100ms Rate**: 74.4%
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

