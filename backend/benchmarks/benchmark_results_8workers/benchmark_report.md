# CareNavigator-AI Performance Benchmark Report

## Executive Summary

- **Total Requests**: 9,000
- **Successful Requests**: 8,000
- **Overall Success Rate**: 88.9%
- **Endpoints Tested**: 9
- **Max Concurrent Users**: 500

## Resume Claims Validation

### Overall Performance
- **Sub-100ms Response Rate**: 41.0%
- **Meets Sub-100ms Claim**: NO
- **Handles 1000+ Concurrent**: NO
- **Max Concurrent Tested**: 500

## Endpoint Performance Details

### health_check
- **Success Rate**: 100.0%
- **Average Response Time**: 234.10ms
- **P95 Response Time**: 484.44ms
- **P99 Response Time**: 529.64ms
- **Sub-100ms Rate**: 21.0%
- **Meets Performance Claims**: NO

### root
- **Success Rate**: 100.0%
- **Average Response Time**: 201.46ms
- **P95 Response Time**: 365.72ms
- **P99 Response Time**: 382.04ms
- **Sub-100ms Rate**: 30.3%
- **Meets Performance Claims**: NO

### metrics
- **Success Rate**: 100.0%
- **Average Response Time**: 143.58ms
- **P95 Response Time**: 365.93ms
- **P99 Response Time**: 411.95ms
- **Sub-100ms Rate**: 58.5%
- **Meets Performance Claims**: NO

### list_models
- **Success Rate**: 100.0%
- **Average Response Time**: 158.67ms
- **P95 Response Time**: 366.88ms
- **P99 Response Time**: 392.58ms
- **Sub-100ms Rate**: 49.1%
- **Meets Performance Claims**: NO

### update_registry
- **Success Rate**: 100.0%
- **Average Response Time**: 173.69ms
- **P95 Response Time**: 403.01ms
- **P99 Response Time**: 432.80ms
- **Sub-100ms Rate**: 46.8%
- **Meets Performance Claims**: NO

### insurance_match
- **Success Rate**: 100.0%
- **Average Response Time**: 193.13ms
- **P95 Response Time**: 427.23ms
- **P99 Response Time**: 495.43ms
- **Sub-100ms Rate**: 34.0%
- **Meets Performance Claims**: NO

### summarize_text
- **Success Rate**: 100.0%
- **Average Response Time**: 208.84ms
- **P95 Response Time**: 427.59ms
- **P99 Response Time**: 485.84ms
- **Sub-100ms Rate**: 37.8%
- **Meets Performance Claims**: NO

### reload_plans
- **Success Rate**: 100.0%
- **Average Response Time**: 153.70ms
- **P95 Response Time**: 390.47ms
- **P99 Response Time**: 461.12ms
- **Sub-100ms Rate**: 50.6%
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

