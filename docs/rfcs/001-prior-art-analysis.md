# RFC 001: Prior Art Analysis - Existing Distributed ML Training Systems

**Status:** ✅ Complete  
**Date:** 2026-04-21  
**Research Task:** Existing Distributed ML Training Systems (Petals, DisTrO, Hivemind)  
**Output:** docs/rfcs/001-prior-art-analysis.md

---

## Executive Summary

This RFC analyzes three major existing distributed ML training systems: **Petals**, **DisTrO**, and **Hivemind**. Each system takes a different approach to distributed training, with distinct tradeoffs in communication patterns, fault tolerance, and use cases. Key findings indicate that decentralized training is feasible but requires careful handling of network latency, straggler mitigation, and gradient aggregation strategies.

---

## 1. Petals

### Overview
Petals is a decentralized system for inference and fine-tuning of large language models (100B+ parameters). It allows volunteers to host parts of models, enabling collaborative inference and fine-tuning without centralized infrastructure.

### Architecture
- **Swarm-based**: Volunteers host "blocks" of transformer layers
- **Adapters/Prompt Tuning**: Fine-tuning updates only prompts or adapters hosted locally
- **No centralized orchestrator**: Uses peer-to-peer discovery and routing

### Gradient Aggregation
- **Focus on fine-tuning**: Primarily designed for inference and adapter/prompt tuning
- **Local updates**: Fine-tuning parameters are updated locally on each node
- **Limited pre-training**: Not designed for full pre-training from scratch

### Task Distribution
- **Layer-wise distribution**: Model layers split across participants
- **Dynamic routing**: Requests routed through swarm to find required blocks
- **Load balancing**: Participants can host multiple blocks based on capacity

### Straggler Mitigation
- **Redundancy**: Multiple copies of popular blocks can exist in swarm
- **Timeout mechanisms**: Unresponsive nodes are skipped
- **Benchmarking**: ~1 step/second for BLOOM-176B on consumer GPUs

### Known Bottlenecks
- **Network latency**: Inference latency depends on network between nodes
- **Cold start**: New nodes need to download model weights
- **Coordination overhead**: Swarm maintenance requires communication

### DistribAI Differences
- Petals focuses on inference/fine-tuning; DistribAI targets full pre-training
- Petals uses P2P swarm; DistribAI uses orchestrator-based architecture
- Petals uses adapters; DistribAI uses full gradient aggregation

### References
- [Petals Research Blog](https://research.yandex.com/blog/petals-decentralized-inference-and-finetuning-of-large-language-models)
- [GitHub Repository](https://github.com/bigscience-workshop/petals)
- [Paper: Petals: Collaborative Inference and Fine-tuning of Large Models](https://arxiv.org/abs/2209.01188)

---

## 2. DisTrO (Distributed Training Over-The-Internet)

### Overview
DisTrO is a family of low-latency distributed optimizers designed specifically for training over the internet. It reduces inter-GPU communication requirements by three to four orders of magnitude compared to traditional distributed training.

### Architecture
- **Optimizer-level innovation**: Reduces communication at the optimizer step
- **Internet-first design**: Built for high-latency, unreliable networks
- **Decentralized training**: Successfully trained 15B and 40B models across internet

### Gradient Aggregation
- **Compressed communication**: Reduces bandwidth needs by 1000-10000x
- **Decentralized optimization**: Each node maintains local optimizer state
- **Asynchronous updates**: Not all nodes need to sync at each step

### Task Distribution
- **Model parallelism**: Model distributed across GPUs
- **Horizontal scaling**: Can add nodes dynamically
- **Proven at scale**: 15B and 40B models trained successfully

### Straggler Mitigation
- **Asynchronous training**: Slow nodes don't block fast nodes
- **Communication reduction**: Less data to transfer reduces straggler impact
- **Fault tolerance**: System can handle node dropout

### Known Bottlenecks
- **Optimizer complexity**: Requires custom optimizer implementation
- **Convergence characteristics**: May differ from standard SGD/Adam
- **Implementation complexity**: Requires deep understanding of optimization theory

### DistribAI Differences
- DisTrO focuses on optimizer-level compression; DistribAI uses gradient compression
- DisTrO is optimizer-agnostic approach; DistribAI may use standard optimizers
- DisTrO has proven production results; DistribAI is in development

### References
- [GitHub Repository](https://github.com/NousResearch/DisTrO)
- [Preliminary Report](https://github.com/NousResearch/DisTrO/raw/main/A_Preliminary_Report_on_DisTrO.pdf)
- [DeMo Optimization Paper](https://arxiv.org/abs/2411.19870)
- [Psyche Network](https://psyche.network/)

---

## 3. Hivemind

### Overview
Hivemind is a decentralized deep learning library in PyTorch designed to train models on thousands of volunteers across the world. It uses a Distributed Hash Table (DHT) for peer discovery and coordination.

### Architecture
- **DHT-based networking**: No master node, fully decentralized
- **Fault-tolerant backpropagation**: Forward/backward passes succeed despite unresponsive nodes
- **PyTorch integration**: Works with existing PyTorch code via Lightning integration

### Gradient Aggregation
- **Decentralized parameter averaging**: Iteratively aggregate updates without full network sync
- **All-reduce alternative**: Uses DHT to coordinate averaging
- **Paper-based approach**: Based on "Decentralized Deep Learning" research

### Task Distribution
- **Mixture-of-Experts**: Parts of layers distributed across participants
- **Arbitrary model size**: Can train models larger than any single node's memory
- **Dynamic allocation**: Participants can join/leave dynamically

### Straggler Mitigation
- **Timeout mechanisms**: Unresponsive nodes are skipped
- **Redundant computation**: Can re-route through alternative paths
- **Asynchronous-friendly**: Not all nodes need to participate in each step

### Known Bottlenecks
- **DHT maintenance**: Overhead for maintaining distributed hash table
- **Convergence speed**: May be slower than centralized all-reduce
- **Complexity**: Requires understanding of decentralized algorithms

### DistribAI Differences
- Hivemind uses DHT for decentralization; DistribAI uses orchestrator
- Hivemind is fully decentralized; DistribAI has centralized coordination
- Hivemind focuses on training; DistribAI includes credits/governance

### References
- [GitHub Repository](https://github.com/learning-at-home/hivemind)
- [Website](https://learning-at-home.github.io/)
- [Paper: Decentralized Deep Learning](https://arxiv.org/abs/2103.03239)
- [Paper: Decentralized Mixture-of-Experts](https://arxiv.org/abs/2002.04013)

---

## Comparative Analysis

| Aspect | Petals | DisTrO | Hivemind | DistribAI |
|--------|--------|--------|----------|----------|
| **Primary Use Case** | Inference + Fine-tuning | Full pre-training | Full training | Full training |
| **Architecture** | P2P Swarm | Optimizer-level | DHT-based | Orchestrator-based |
| **Gradient Aggregation** | Local adapters | Compressed optimizer | Decentralized averaging | Centralized orchestrator |
| **Fault Tolerance** | Timeout + redundancy | Asynchronous | Fault-tolerant backprop | Re-assignment |
| **Network Requirements** | Moderate | Optimized for internet | Optimized for internet | Standard |
| **Proven Scale** | 176B inference | 40B training | Various models | TBD (mock only) |
| **Decentralization** | High | High | Very High | Medium (orchestrator) |
| **Credits/Governance** | None | None | None | Planned |

---

## Key Lessons for DistribAI

### 1. Communication is the Bottleneck
- All three systems prioritize reducing communication overhead
- DisTrO's 1000-10000x reduction is particularly compelling
- Gradient compression (RFC 002) should be a priority

### 2. Fault Tolerance is Essential
- Internet-scale training must handle node failures gracefully
- Timeout mechanisms and redundancy are standard
- DistribAI's re-assignment strategy aligns with this

### 3. Straggler Mitigation Required
- Asynchronous updates help (DisTrO, Hivemind)
- Redundancy helps (Petals)
- DistribAI should implement both

### 4. Tradeoff: Centralization vs Decentralization
- Petals/Hivemind: Fully decentralized, harder to coordinate
- DistribAI: Orchestrator-based, easier coordination but single point of failure
- Consider hybrid approach: multiple orchestrators with DHT discovery

### 5. Gradient Aggregation Strategy
- Petals: Local-only (not applicable for pre-training)
- DisTrO: Optimizer-level compression
- Hivemind: Decentralized averaging
- DistribAI: Centralized with compression (need to validate this choice)

---

## Recommendations

### Immediate Actions
1. **Study DisTrO's optimizer approach**: Consider integrating similar communication reduction techniques
2. **Implement gradient compression**: Priority per RFC 002
3. **Add timeout mechanisms**: Borrow from Petals/Hivemind for fault tolerance
4. **Consider hybrid architecture**: Evaluate if full decentralization (Hivemind-style) is worth the complexity

### Future Considerations
1. **Benchmark against Petals**: Test DistribAI's inference capabilities
2. **Evaluate DHT for orchestrator discovery**: Could add resilience
3. **Study DisTrO's convergence characteristics**: May inform optimizer choice
4. **Monitor Hivemind developments**: Active project with ongoing research

---

## Open Questions

1. **Orchestrator scalability**: Can a single orchestrator handle 1000+ nodes, or do we need multiple?
2. **Gradient compression vs optimizer compression**: Should DistribAI use DisTrO-style optimizer compression or traditional gradient compression?
3. **Centralization tradeoff**: Is the orchestrator's single point of failure acceptable, or should we adopt Hivemind's fully decentralized approach?
4. **Inference vs training**: Should DistribAI support Petals-style inference, or focus solely on training?

---

## Conclusion

All three systems demonstrate that distributed ML training over the internet is feasible, but each takes a different approach. Petals excels at inference/fine-tuning with minimal coordination. DisTrO achieves remarkable communication reduction through optimizer innovation. Hivemind provides a fully decentralized training framework.

DistribAI's orchestrator-based architecture sits between these approaches - more centralized than Hivemind/Petals but with simpler coordination. The key differentiator will be DistribAI's credit system and governance, which none of these systems address.

**Next step:** Proceed to RFC 002 on Gradient Compression Strategies to determine the best approach for reducing communication overhead in DistribAI.
