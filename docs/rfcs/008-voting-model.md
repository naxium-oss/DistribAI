# RFC 008: Quadratic Voting vs Flat Voting Analysis

**Status:** Completed  
**Date:** 2026-04-21  
**Research Lead:** DistribAI Research Team

---

## Summary

Research on voting model tradeoffs for DistribAI governance. Analyzed quadratic voting vs flat 1-credit-1-vote systems to determine optimal community participation model.

## Key Findings

### Quadratic Voting (QV) Benefits
- **Reduced whale dominance**: QV imposes increasing marginal cost on large votes (cost = votes²)
- **More democratic**: Small contributors have disproportionately more voice per credit
- **Collusion resistant**: More expensive to coordinate large voting blocs

### QV Implementation
```
cost_to_vote = votes^2
credits_needed = 1, 4, 9, 16, 25, 36, 49, 64, 81, 100 for 1-10 votes
```

### Flat Voting Benefits
- **Simple to understand**: 1 credit = 1 vote
- **No computational overhead**: Direct credit deduction
- **Predictable**: Users know exactly what they're spending

### Research Conclusions

| Metric | Flat | Quadratic |
|--------|------|-----------|
| Whale deterrence | Low | High |
| Small user participation | Medium | High |
| UX complexity | Low | Medium |
| Sybil resistance | Medium | High |

## Recommendation for DistribAI

**Phase 3 (Launch): Flat voting with velocity caps**
- 1 credit = 1 vote
- Maximum votes per account per hour: 1000
- Maximum votes per job per account: 500

**Phase 5 (Ecosystem): Evaluate QV transition**
- Monitor Gitcoin quadratic funding results
- Survey community on QV preference
- Implement if 70%+ support

## References

1. Dimitri, N. "Quadratic Voting in Blockchain Governance." Information 2022, 13, 305.
2. Weyl, E.G. "The Robustness of Quadratic Voting." Public Choice 2017.
3. Gitcoin Quadratic Funding Analysis (2023)

---
*End of RFC 008*
