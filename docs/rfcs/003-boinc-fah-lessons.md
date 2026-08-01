# RFC 003: BOINC / Folding@Home Architecture Deep Dive

**Status:** ✅ Complete  
**Date:** 2026-04-21  
**Research Task:** BOINC / Folding@Home Architecture  
**Output:** docs/rfcs/003-boinc-fah-lessons.md

---

## Executive Summary

This RFC analyzes the architecture of BOINC (Berkeley Open Infrastructure for Network Computing) and Folding@home, two of the most successful volunteer computing platforms with 20+ years of operation. Key findings highlight the importance of redundant computation for validation, benchmark-based credit systems, and sophisticated scheduling for heterogeneous hardware. These lessons are directly applicable to DistribAI's design for task distribution, credit systems, and cheating prevention.

---

## 1. BOINC Architecture

### Overview
BOINC is a middleware system for volunteer computing that enables projects to distribute jobs to volunteer hosts worldwide. It consists of a server system and client software that communicate to distribute, process, and return workunits.

### Server Structure
BOINC servers run on Linux-based computers using Apache, PHP, and MySQL for web and database systems. The server consists of:

- **Two CGI programs**: Scheduler and data server
- **Five daemons**: Feeder, validator, assimilator, file deleter, transitioner

#### Key Components

**Scheduler CGI**
- Handles client requests for work
- Receives completed results
- Sends new work to compute
- Reads from shared-memory block populated by feeder daemon

**Feeder Daemon**
- Loads tasks from database
- Keeps tasks in shared-memory block
- Periodically refills empty slots after scheduler sends results

**Validator**
- Examines instances of a job
- Compares output files
- Decides whether quorum of equivalent results exists
- Marks work unit and valid results as valid
- Chooses "canonical result"
- Grants credit to users with legitimate results

**Assimilator**
- Processes canonical result using project-specific code
- May parse files and store in database
- May copy files elsewhere
- Can generate more workunits based on returned data

**File Deleter**
- Deletes output files after assimilator processes them
- Deletes input files no longer needed

**Transitioner**
- Handles state transitions of workunits and results
- Generates results from workunits when first created
- Generates additional results when needed (e.g., invalid result)

### Task Distribution

#### Workunit and Result Model
- **Workunit**: Computation to be performed by clients
- **Result**: Instance of a workunit (even if not completed)
- Server automatically creates results from workunits (not explicit by project)

#### Scheduling Features
- **Homogeneous redundancy**: Send workunits only to computers of same platform (e.g., Win XP SP2 only)
- **Workunit trickling**: Send information to server before workunit completes
- **Locality scheduling**: Send workunits to computers that already have necessary files; create work on demand
- **Host-based distribution**: Workunits requiring 512 MB RAM only sent to hosts with at least that much RAM

#### Redundant Computation (Job Replication)
- Each computation performed on multiple clients
- Results compared and accepted only when consensus reached
- If results don't agree, or if one result not reported by deadline, server generates additional instance
- Repeated until quorum of matching results found or limit on instances reached

### Validation Mechanism

#### Quorum-Based Validation
1. Wait for N results from different hosts (N configurable, typically 2-5)
2. Compare results using project-specific validation:
   - Bitwise comparison for deterministic computations
   - Fuzzy comparison for floating-point or stochastic computations
3. If quorum of matching results found:
   - Mark work unit as valid
   - Mark valid results as valid
   - Choose canonical result
   - Grant credit to users with valid results
4. If no quorum:
   - Generate additional result instance
   - Send to new host
   - Repeat until quorum or max instances reached

#### Credit System
- **Credit formula**: Based on benchmarked performance
- **Computation credit**: Calculated based on floating-point operations (FLOPs)
- **Benchmark tests**: Simple math (addition, multiplication, division) and trigonometric/exponential functions
- **Credit granularity**: Credit awarded when work unit validated and marked valid

### Heterogeneous Hardware Support

#### Platform Detection
- Client reports host parameters: CPU type, RAM, OS, floating-point capabilities
- Server uses this information for scheduling decisions
- Homogeneous redundancy ensures results comparable across similar platforms

#### Scheduling Policies
- Work distribution based on host parameters (RAM, CPU speed, disk space)
- Threshold-based scheduling policies for efficiency
- Local scheduling at host level addresses:
  - When to fetch work
  - Which work to run first
  - When to report results

---

## 2. Folding@Home Architecture

### Overview
Folding@home (FAH) is a distributed computing project for simulating protein dynamics. It uses a custom client architecture rather than BOINC, but shares many design principles.

### Credit System

#### Reference Machine Benchmarking
FAH uses a single benchmark machine to determine points for work units:

**Benchmark Machine Specs:**
- Processor: Intel(R) Core(TM) i5 CPU 750 @ 2.67GHz
- OS: Linux

#### Point Calculation Process
1. Take a work unit (WU) from a project
2. Run it on benchmark machine until completion
3. Measure completion time
4. Base credit = scaling factor × completion time
5. Set timeout and deadline values as functions of completion time
6. Apply k-factor (coefficient for bonus points, baseline 0.75)

#### Final Point Formula
```
final_points = base_points × max(1, sqrt(k × deadline_length / elapsed_time))
```

Where:
- `max(1, ...)` ensures final_points never lower than base_points
- `deadline_length` is the final deadline
- `elapsed_time` is time from assignment to upload (including transit)
- Both measured in days to one decimal point

#### Points Per Day (PPD) Calculation
```
PPD = 14.4 × base_points × max(1, sqrt(14.4 × k × Expiration / TPF)) / TPF
```

Where:
- `TPF` is time per frame in minutes (decimal form)
- `Expiration` is project-specific expiration time

#### GPU Benchmarking
- GPU projects benchmarked on same machine using CPU
- Unified benchmarking scheme for both GPU and CPU projects
- Goal: "equal pay for equal work" using same yardstick (i5 benchmark CPU)
- GPU FAHCore 17 enables running any CPU computation on GPU

### Validation and Cheating Prevention

#### Benchmark Consistency
- Reference machine ensures consistency across work units
- Natural variation between machines and WUs prevents perfect reflection
- Goal: Consistency within reference machine definition, not absolute accuracy

#### Deadline Enforcement
- Deadlines set based on benchmark machine completion time
- Gives donors reasonable time to finish
- Short enough to allow retrieval and reassignment of unprocessed WUs
- Varies by hardware type (uniprocessor, SMP, GPU)

---

## 3. Key Lessons for DistribAI

### Task Distribution

#### Lesson 1: Redundant Computation is Essential
- **BOINC approach**: Send each job to multiple hosts, compare results
- **DistribAI application**: For critical training steps, use redundant computation
- **Implementation**: 
  - For gradient aggregation, require 2-3 nodes to compute same batch
  - Compare gradients; use majority or weighted average
  - Detect and reject anomalous results (potential Byzantine nodes)

#### Lesson 2: Homogeneous Redundancy Improves Validation
- **BOINC approach**: Send work only to similar platforms
- **DistribAI application**: Group nodes by hardware tier before assignment
- **Implementation**:
  - Benchmark nodes on registration (CPU, GPU, memory, network)
  - Assign to tiers (e.g., "high-end GPU", "mid-range GPU", "CPU-only")
  - For redundant computation, use nodes from same tier
  - Simplifies gradient comparison (similar precision characteristics)

#### Lesson 3: Locality Scheduling Reduces Bandwidth
- **BOINC approach**: Send work to hosts that already have necessary files
- **DistribAI application**: Cache model weights and data on nodes
- **Implementation**:
  - Nodes download model weights on first assignment
  - Subsequent jobs use cached weights
  - Only transmit deltas/updates for model changes
  - Significant bandwidth savings for large models

### Credit System

#### Lesson 4: Benchmark-Based Credits Ensure Fairness
- **FAH approach**: Single reference machine for all benchmarking
- **DistribAI application**: Define reference hardware for credit calculation
- **Implementation**:
  - Define reference machine (e.g., RTX 3080, 32GB RAM, 1Gbps network)
  - Benchmark each training job on reference machine
  - Calculate base credits = scaling factor × benchmark time
  - Adjust for actual node performance (faster = more credits/time)
  - Formula: `credits = base_credits × (reference_time / actual_time)`

#### Lesson 5: Deadline-Based Bonus Incentivizes Reliability
- **FAH approach**: Bonus points for early completion
- **DistribAI application**: Reward nodes that complete jobs quickly
- **Implementation**:
  - Set job deadline based on benchmark time + safety margin
  - Apply bonus multiplier for early completion
  - Formula: `final_credits = base_credits × max(1, sqrt(k × deadline / elapsed))`
  - Encourages nodes to stay online and responsive

#### Lesson 6: K-Factor for Project Importance
- **FAH approach**: Adjust k-factor based on scientific value
- **DistribAI application**: Higher credits for more important training jobs
- **Implementation**:
  - Base k-factor = 1.0
  - Adjust for job priority (critical path jobs = 1.5x)
  - Adjust for data importance (high-value datasets = 1.2x)
  - Adjust for model stage (pre-training vs fine-tuning)

### Cheating Prevention

#### Lesson 7: Quorum Validation Detects Malicious Nodes
- **BOINC approach**: Require quorum of matching results
- **DistribAI application**: Validate gradients from multiple nodes
- **Implementation**:
  - For each training step, assign batch to 2-3 nodes
  - Compare gradients; require agreement within tolerance
  - Use statistical methods to detect outliers (see RFC 006)
  - Reject results from nodes that consistently disagree

#### Lesson 8: Deadline Enforcement Prevents Stalling
- **BOINC/FAH approach**: Strict deadlines with reassignment
- **DistribAI application**: Time-bound job assignments
- **Implementation**:
  - Set deadlines based on benchmark + network latency
  - Reassign jobs if node misses deadline
  - Penalize nodes with high timeout rates
  - Ban nodes with chronic reliability issues

#### Lesson 9: Canonical Result Selection
- **BOINC approach**: Choose one valid result as canonical
- **DistribAI application**: Aggregate gradients from valid nodes
- **Implementation**:
  - If 2/3 nodes agree, use those gradients
  - If all disagree, flag for manual review or discard
  - Track node reliability scores over time
  - Weight node contributions by reliability

### Heterogeneous Hardware

#### Lesson 10: Host-Based Scheduling
- **BOINC approach**: Match work to host capabilities
- **DistribAI application**: Assign jobs based on node tier
- **Implementation**:
  - Benchmark nodes on registration
  - Maintain hardware profiles
  - Large models → high-tier nodes only
  - Small models → any tier
  - Adjust batch size based on node capabilities

#### Lesson 11: Platform-Specific Optimizations
- **BOINC approach**: Separate binaries for different platforms
- **DistribAI application**: Optimize for different GPU architectures
- **Implementation**:
  - Compile CUDA kernels for different GPU generations
  - Detect GPU architecture on node registration
  - Send appropriate binaries
  - Enable architecture-specific optimizations (Tensor Cores, etc.)

---

## 4. Recommended Architecture for DistribAI

### Server Components (Inspired by BOINC)

```
Orchestrator (DistribAI equivalent to BOINC server)
├── Job Scheduler (equivalent to BOINC scheduler)
│   ├── Assigns jobs to nodes based on tier and availability
│   ├── Handles job completion and reassignment
│   └── Manages job queues and priorities
├── Validator (new component for gradient validation)
│   ├── Compares gradients from redundant computation
│   ├── Detects Byzantine behavior (RFC 006)
│   └── Aggregates valid gradients
├── Credit Manager (new component for credit calculation)
│   ├── Maintains reference benchmarks
│   ├── Calculates credits based on performance
│   └── Applies deadline bonuses
├── Node Manager (enhanced version of BOINC feeder)
│   ├── Tracks node status and reliability
│   ├── Manages node tier assignments
│   └── Handles node registration/deregistration
└── Storage Manager (equivalent to BOINC assimilator)
    ├── Aggregates training results
    ├── Manages model checkpoints
    └── Handles dataset distribution
```

### Client Components (DistribAI Worker)

```
Worker (DistribAI equivalent to BOINC client)
├── Job Executor
│   ├── Downloads model weights and data
│   ├── Executes training steps
│   ├── Uploads gradients
│   └── Manages local caching
├── Benchmark Module
│   ├── Runs benchmarks on registration
│   ├── Reports hardware capabilities
│   └── Updates performance metrics
├── Credit Tracker
│   ├── Tracks completed jobs
│   ├── Calculates earned credits
│   └── Reports to orchestrator
└── Health Monitor
    ├── Reports heartbeat
    ├── Monitors resource usage
    └── Handles graceful shutdown
```

### Validation Strategy

#### Phase 0 (Development - 10 nodes)
- No redundant computation (single node per batch)
- Basic credit tracking (time-based)
- No Byzantine detection
- Focus on functional correctness

#### Phase 1 (Alpha - 50 nodes)
- Redundant computation for 10% of batches (2 nodes)
- Statistical gradient comparison
- Basic reliability scoring
- Deadline enforcement with reassignment

#### Phase 2 (Beta - 200 nodes)
- Redundant computation for 50% of batches (2-3 nodes)
- Byzantine detection (RFC 006)
- Credit system with benchmark-based calculation
- Tier-based scheduling

#### Phase 3+ (Production - 1000+ nodes)
- Redundant computation for 100% of batches (3 nodes)
- Advanced Byzantine detection
- Sophisticated credit system with bonuses
- Full heterogeneous hardware support

---

## 5. Open Questions

1. **Reference Machine Selection**: What should DistribAI's reference machine be? RTX 3080? A100? Cloud instance?
2. **Redundancy Ratio**: How many nodes per batch for redundant computation? 2? 3? Dynamic based on node reliability?
3. **Credit Scaling**: How to translate compute time to meaningful credits? Should credits be convertible to tokens?
4. **Gradient Comparison Tolerance**: What tolerance for gradient comparison? How to handle floating-point precision differences?
5. **Benchmark Frequency**: How often to re-benchmark nodes? On every registration? Periodically?
6. **Deadline Calculation**: What safety margin over benchmark time? 2x? 3x? Dynamic based on network conditions?

---

## 6. Implementation Priority

### High Priority (Phase 0-1)
1. **Node benchmarking**: Implement hardware detection and benchmarking
2. **Deadline enforcement**: Add timeouts and reassignment logic
3. **Basic credit tracking**: Time-based credit calculation
4. **Tier-based scheduling**: Simple hardware tier assignment

### Medium Priority (Phase 1-2)
1. **Redundant computation**: Implement 2-node validation for critical batches
2. **Gradient validation**: Statistical comparison of gradients
3. **Benchmark-based credits**: Reference machine and credit formula
4. **Reliability scoring**: Track node success/failure rates

### Low Priority (Phase 2-3)
1. **Advanced Byzantine detection**: Full implementation per RFC 006
2. **Bonus credit system**: Deadline-based multipliers
3. **Locality scheduling**: Model weight caching
4. **Homogeneous redundancy**: Platform-specific validation

---

## Conclusion

BOINC and Folding@home provide 20+ years of battle-tested lessons in volunteer computing architecture. The key takeaways for DistribAI are:

1. **Redundant computation with quorum validation** is essential for reliability and cheating prevention
2. **Benchmark-based credit systems** ensure fairness across heterogeneous hardware
3. **Deadline enforcement** prevents stalling and incentivizes reliability
4. **Tier-based scheduling** matches work to hardware capabilities
5. **Locality scheduling** reduces bandwidth through caching

DistribAI should adopt a hybrid approach: BOINC-style redundant computation for validation, Folding@home-style benchmark-based credits, and a custom orchestrator architecture optimized for ML training workloads.

**Next step:** Proceed to RFC 004 on Client UI Stack Decision (Tauri vs Electron) to inform the worker client implementation.
