# RFC 009: Secure Aggregation for Gradient Privacy

**Status:** Completed  
**Date:** 2026-04-21  
**Research Lead:** DistribAI Security Team

---

## Summary

Analysis of secure aggregation protocols for privacy-preserving distributed training. Evaluates Google's Secure Aggregation protocol (Bonawitz et al.) and FATE framework for DistribAI Phase 4 implementation.

## Secure Aggregation Overview

### Threat Model
- **Honest-but-curious server**: Follows protocol but tries to learn individual gradients
- **Colluding workers**: Up to f < n/2 workers sharing information
- **External eavesdroppers**: Network-level attackers

### Google Secure Aggregation Protocol (Bonawitz et al. 2017)

**Core Mechanism:**
1. Each worker masks gradient with pairwise random seeds
2. Server aggregates masked gradients (masks cancel out)
3. Result: server learns only aggregate, not individual contributions

**Key Properties:**
- Computes sum without revealing individual values
- Uses Diffie-Hellman key exchange for pairwise masking
- Robust to dropouts (t-out-of-n threshold)
- Communication: O(n) per round for n workers

**Performance Overhead:**
- Computation: ~2x gradient computation time
- Communication: ~3x bandwidth for mask exchange
- Latency: +1 RTT for key establishment

### FATE Framework (WeBank)

**Components:**
- **Secure Aggregation Module**: Homomorphic encryption option
- **Differential Privacy**: Optional noise addition
- **PSI (Private Set Intersection)**: For participant validation

**Tradeoffs:**
- More flexible than Google's protocol
- Higher computational overhead (homomorphic ops)
- Better for heterogeneous networks

## DistribAI Recommendation

### Phase 4 Implementation: Google's Protocol

**Rationale:**
- Battle-tested at Google scale (billions of devices)
- Lower overhead than homomorphic alternatives
- Compatible with Byzantine detection (can inspect aggregates)

**Integration Points:**
```python
# Before gradient upload
masked_grad = secure_agg.mask_gradient(gradient, peer_seeds)

# Server aggregates
aggregate = secure_agg.aggregate_masked(masked_grads)

# Result is plaintext aggregate
decrypted = secure_agg.unmask_aggregate(aggregate)
```

**Conflict Resolution:**
- Secure aggregation prevents individual gradient inspection
- BUT: DistribAI needs Byzantine detection
- Solution: Apply Byzantine detection to aggregates only
- Accept reduced granularity for privacy gains

## References

1. Bonawitz, K. et al. "Practical Secure Aggregation for Federated Learning." CCS 2017.
2. FATE Framework Documentation (WeBank AI)
3. "Secure Multi-Party Computation for Federated Learning" - NeurIPS 2022 Workshop

---
*End of RFC 009*
