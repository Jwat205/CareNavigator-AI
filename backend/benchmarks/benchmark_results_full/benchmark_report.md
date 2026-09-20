# CareNavigator-AI Performance Benchmark Report

## Executive Summary

- **Total Requests**: 22,500
- **Successful Requests**: 20,000
- **Overall Success Rate**: 88.9%
- **Endpoints Tested**: 9
- **Max Concurrent Users**: 500

## Resume Claims Validation

### Overall Performance
- **Sub-100ms Response Rate**: 58.5%
- **Meets Sub-100ms Claim**: NO
- **Handles 1000+ Concurrent**: NO
- **Max Concurrent Tested**: 500

## Endpoint Performance Details

### health_check
- **Success Rate**: 100.0%
- **Average Response Time**: 114.62ms
- **P95 Response Time**: 494.69ms
- **P99 Response Time**: 584.12ms
- **Sub-100ms Rate**: 65.1%
- **Meets Performance Claims**: NO

### root
- **Success Rate**: 100.0%
- **Average Response Time**: 108.67ms
- **P95 Response Time**: 516.62ms
- **P99 Response Time**: 614.18ms
- **Sub-100ms Rate**: 72.4%
- **Meets Performance Claims**: NO

### metrics
- **Success Rate**: 100.0%
- **Average Response Time**: 140.19ms
- **P95 Response Time**: 622.50ms
- **P99 Response Time**: 736.83ms
- **Sub-100ms Rate**: 63.5%
- **Meets Performance Claims**: NO

### list_models
- **Success Rate**: 100.0%
- **Average Response Time**: 124.48ms
- **P95 Response Time**: 530.31ms
- **P99 Response Time**: 625.08ms
- **Sub-100ms Rate**: 60.3%
- **Meets Performance Claims**: NO

### update_registry
- **Success Rate**: 100.0%
- **Average Response Time**: 136.88ms
- **P95 Response Time**: 551.53ms
- **P99 Response Time**: 659.49ms
- **Sub-100ms Rate**: 59.6%
- **Meets Performance Claims**: NO

### insurance_match
- **Success Rate**: 100.0%
- **Average Response Time**: 183.72ms
- **P95 Response Time**: 820.16ms
- **P99 Response Time**: 950.52ms
- **Sub-100ms Rate**: 47.5%
- **Meets Performance Claims**: NO

### summarize_text
- **Success Rate**: 100.0%
- **Average Response Time**: 166.19ms
- **P95 Response Time**: 659.70ms
- **P99 Response Time**: 954.85ms
- **Sub-100ms Rate**: 57.3%
- **Meets Performance Claims**: NO

### reload_plans
- **Success Rate**: 100.0%
- **Average Response Time**: 214.85ms
- **P95 Response Time**: 1086.06ms
- **P99 Response Time**: 1272.51ms
- **Sub-100ms Rate**: 41.9%
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

