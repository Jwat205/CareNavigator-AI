# CareNavigator-AI Performance Benchmark Report

## Executive Summary

- **Total Requests**: 22,500
- **Successful Requests**: 20,000
- **Overall Success Rate**: 88.9%
- **Endpoints Tested**: 9
- **Max Concurrent Users**: 500

## Resume Claims Validation

### Overall Performance
- **Sub-100ms Response Rate**: 74.4%
- **Meets Sub-100ms Claim**: NO
- **Handles 1000+ Concurrent**: NO
- **Max Concurrent Tested**: 500

## Endpoint Performance Details

### health_check
- **Success Rate**: 100.0%
- **Average Response Time**: 134.80ms
- **P95 Response Time**: 485.45ms
- **P99 Response Time**: 532.00ms
- **Sub-100ms Rate**: 71.6%
- **Meets Performance Claims**: NO

### root
- **Success Rate**: 100.0%
- **Average Response Time**: 132.84ms
- **P95 Response Time**: 471.23ms
- **P99 Response Time**: 525.10ms
- **Sub-100ms Rate**: 75.3%
- **Meets Performance Claims**: NO

### metrics
- **Success Rate**: 100.0%
- **Average Response Time**: 108.99ms
- **P95 Response Time**: 300.09ms
- **P99 Response Time**: 355.27ms
- **Sub-100ms Rate**: 73.8%
- **Meets Performance Claims**: NO

### list_models
- **Success Rate**: 100.0%
- **Average Response Time**: 121.28ms
- **P95 Response Time**: 407.50ms
- **P99 Response Time**: 456.13ms
- **Sub-100ms Rate**: 75.6%
- **Meets Performance Claims**: NO

### update_registry
- **Success Rate**: 100.0%
- **Average Response Time**: 107.32ms
- **P95 Response Time**: 324.16ms
- **P99 Response Time**: 375.81ms
- **Sub-100ms Rate**: 77.1%
- **Meets Performance Claims**: NO

### insurance_match
- **Success Rate**: 100.0%
- **Average Response Time**: 132.20ms
- **P95 Response Time**: 399.69ms
- **P99 Response Time**: 450.51ms
- **Sub-100ms Rate**: 72.5%
- **Meets Performance Claims**: NO

### summarize_text
- **Success Rate**: 100.0%
- **Average Response Time**: 146.02ms
- **P95 Response Time**: 534.24ms
- **P99 Response Time**: 602.39ms
- **Sub-100ms Rate**: 73.6%
- **Meets Performance Claims**: NO

### reload_plans
- **Success Rate**: 100.0%
- **Average Response Time**: 121.36ms
- **P95 Response Time**: 392.18ms
- **P99 Response Time**: 448.78ms
- **Sub-100ms Rate**: 75.4%
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

