# Smoke Test Report

**Total:** 12  |  **Passed:** 12  |  **Failed:** 0


| # | Test | Status | Time (ms) |
|---|------|--------|-----------|
| 1 | Core imports | ✅ PASS | 74.1 |
| 2 | Cache hit/miss | ✅ PASS | 0.0 |
| 3 | Capacity enforcement | ✅ PASS | 0.2 |
| 4 | Urgency mathematics | ✅ PASS | 0.0 |
| 5 | Popularity window | ✅ PASS | 0.0 |
| 6 | Baseline eviction | ✅ PASS | 0.0 |
| 7 | Content catalog | ✅ PASS | 10.3 |
| 8 | Highway simulation | ✅ PASS | 1.0 |
| 9 | Simulation runner | ✅ PASS | 15.4 |
| 10 | Full benchmark | ✅ PASS | 19.7 |
| 11 | Config round-trip | ✅ PASS | 18.1 |
| 12 | Metrics & ranking | ✅ PASS | 9.1 |

## Environment

- Python: 3.12.3
- numpy: available
- pyyaml: available
- fastapi/pytest: **not available** (network-isolated container)

## Notes

- API tests skipped (fastapi not installable in offline container)
- All core cache/simulation/evaluation logic validated