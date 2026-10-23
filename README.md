# DeepSeek-V4.1-Flash DGX Spark / GB10 and x86 RTX 3090 offload: measured findings

These findings explain the costs, failures and limits of moving Engram lookups and speculative drafting off two DGX Spark nodes. The tables distinguish direct measurements, offline analysis, and engineering estimates. The [full-stack overview](https://github.com/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload) supplies architecture and the baseline comparison.

**Hugging Face:** [Card and files](https://huggingface.co/ProprietaryLegal/dsv41-spark-offload-findings) · **Stack overview:** [GitHub](https://github.com/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload) · **Collection:** [All entries](https://huggingface.co/collections/ProprietaryLegal/deepseek-v41-flash-on-2x-dgx-spark-x86-offload-6adab604a517b426ded3916e)

<!-- pli:blurb:start -->

> Published by [PLI Labs](https://proprietarylegal.ai/labs.html), the open research arm of [Proprietary Legal Intelligence](https://proprietarylegal.ai). PLI builds AI systems around a legal organization's own work product, combining retrieval, staged workflows, validation, and attorney review. Visit [proprietarylegal.ai](https://proprietarylegal.ai) to explore our work and discuss what your organization needs.

<!-- pli:blurb:end -->

## Status

2026-10-23 — research-only findings, backed by deployed Engram and one-card async drafting; the two-card runtime is opt-in with promotion pending, while native fidelity and the additional cache/scheduler features remain unqualified.

Later reviews clear the two-card ownership fixes for integration and prefix-storage code for gated testing. The prefill-ordering liveness fix is implemented and CPU-tested; fresh independent review and GB10 validation are pending.

## What it is

1. **Remote round cost:** the principal explanation is lost CPU/GPU overlap, not cable serialization. The separate-process handoff restores overlap without the thread proxy's prefill cost.
2. **GB10 async safety:** parking a stream while a proxy makes CUDA calls can deadlock module loading. A CUDA-free host-flag helper and separate torch-free proxy process avoid that dependency. Late proxy completion also needs private headers and resync buffers to prevent silent draft corruption.
3. **Shared fabric:** bulk traffic on the host links can contend with the Spark TP fabric. Normal offload traffic is much smaller.
4. **Failure behavior:** worker outages fall back to target-only generation; proxy loss can trigger bounded respawn. Recovery must preserve request state and GPU ownership.
5. **Exactness methodology:** the base engine itself is nondeterministic at long context. Boundary comparisons, matched-prefix diagnostics and direct step timers answer different questions and should not be conflated.
6. **Negative results:** link speed, CPU pinning, module-loading policy and a shorter interpreter interval did not solve the synchronous round cost or thread-proxy prefill penalty on their own.
7. **NVLink memory safety:** peer buffers must sit outside Torch expandable segments, and every graph capture must own a stream on its intended device.
8. **Acceptance headroom:** exact target-logprob replay gives a substantially higher target-greedy ceiling than the early logit lens. The [standalone method](https://github.com/ProprietaryLegal/dsv41-exact-acceptance-ceiling) contains the detailed estimator, capture format and aggregates.
9. **Open work:** packed KV, larger context, prefix storage, prefill ordering, native fidelity, self-distillation and further upstream adoption have separate qualification gates.

## Why

Offload changes memory placement, synchronization, scheduling and failure recovery together. A fast worker is not sufficient if the engine CPU waits for it at the wrong point. More cache capacity helps only when the workload uses it; a numerical proxy or allocator calculation is not an end-to-end performance result.

### Engineering estimates, not measured link upgrades

Conditions: arithmetic reported 2026-10-22 for one worst-case single-request payload and the measured hardware configuration; no new cable or projected-state deployment was tested.

| idea | estimate and assumptions |
|---|---|
| faster host cable | a six-row, 184,416-byte request serializes in 36.9 µs at 40 Gb/s versus 14.8 µs at 100 Gb/s; at most about 0.022 ms saved per 42.7 ms round, about 0.05% |
| link occupancy | one such request per 42.7 ms is about 0.09% of 40 Gb/s capacity; protocol overhead and larger transfers are separate |
| projected conditioning instead of taps | about 25 µs wire saving, at the cost of about 47 MiB of projection weights and projection work on the Spark |

These estimates are too small to explain the measured millisecond-scale single-stream penalty. They are not predictions for bulk transfer workloads.

## When to use it

Use these findings to plan controlled offload comparisons, choose where to instrument the critical path, and understand which fallback and numerical checks need to pass before enabling an optimization. Reproduce with your own target and prompts.

## When not to use it

Do not treat this page as an automatic installer, an assurance of deterministic output, or a qualified result for native drafting or unfinished cache/scheduler changes. CPU loopback does not traverse a physical Spark cable. Binned histogram percentiles from separate scopes cannot be subtracted to measure a missing phase.

## Requirements

The measured system uses two DGX Spark / GB10 nodes plus an x86 RAM offload host with RTX 3090 drafting, RDMA links, and the same [public EXL3 target](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-LegalWeighted-DGX-Spark). Consult the [overview](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload) for the hardware/software pins.

The loopback source needs a C++ compiler and separately installed ofl header/library. The acceptance scorer needs CPU Python, PyTorch, safetensors, private captures and reference data. This findings copy does not include all scorer dependencies; use the complete method repository for reproduction.

## Measured effect

### 1. Synchronous remote round cost and passive observation

Conditions: 2026-10-22, two GB10 target nodes and one x86 RTX 3090 drafter, same target and `T = 1`; stock local drafter versus remote synchronous capacity mode without spending the freed memory on KV. The normalized statistic uses accepted drafts and delivered tokens and is diagnostic rather than a direct step trace.

| workload; one concurrent request | added time/round, ms | change | repeats per configuration |
|---|---:|---:|---:|
| short | 2.724 | +7.32% | 3 |
| 32K cold | 2.451 | +6.10% | 2 |
| 150K cold | 3.343 | +8.48% | 2 |

| timing / observation | value | conditions and limitations |
|---|---|---|
| remote worker / local GB10 proposal p50 | 3.444 / 4.467 ms | 2026-10-22; remote capacity histogram across 6,144 rounds; local shadow proposal histogram, 1,023 timed observations; different scopes, upper edges, no direct subtraction |
| outside-worker mean RPC residual | 0.146 ms | nested means from 1,024 shadow rounds; includes transport, callbacks and encoding |
| outside-gather/RPC mean residual | 0.152 ms | same 1,024-round shadow population; bookkeeping and enqueue work, not a packing-only timer |
| connection-thread runqueue delay | 2.596 ms across 262 worker steps, about 0.010 ms/step | one passive active/idle window on x86 host, 2026-10-22 |
| host idle / observed PCIe traffic | 94–96% idle; at most 23 MB/s observed device traffic | two passive windows, 2026-10-22; no swap/reclaim growth; IRQ changes negligible; governor already `performance` |

Draft compute itself is fast remotely. The synchronous path waits for target verification, network and worker before returning to the engine CPU, losing scheduling overlap that the local queued proposal permits. Passive samples do not show pressure sufficient to explain the penalty. Other jobs used separate GPUs; this is bounded evidence, not an exclusion of every possible driver or memory-bandwidth interaction.

### 2. Async deadlocks, placement and hardening

`cuStreamWaitValue32` can park the engine stream ahead of a proxy's CUDA module load. Eager module loading covers lazy loads but not explicit Triton module loading. The working path uses host-mapped counters and a proxy that makes no CUDA calls.

Conditions: 2026-10-23, same two GB10 target nodes and x86 worker; matched short/32K/150K workloads, `T = 1`, 1,500-token budgets. “Three” and “two” below count sequential requests unless concurrency is stated.

| variant / effect | observation | repeats / scope |
|---|---|---|
| initial in-process async proxy | lower step p50; TTFT +4% at 32K, +5–8% at 150K | three short/32K requests, two 150K requests; one stream |
| same initial proxy, watchdog | 31 fires at a 200 ms deadline | one suite; repeated with default interpreter interval |
| default interpreter interval | 5 ms did not remove the TTFT cost | same suite repeated, not an independent performance gain |
| eager module loading | startup 1,456 s versus 364 s; synchronous TTFT normal | fresh-start controls; about 18 minutes extra; did not explain async TTFT cost |
| one-card separate process proxy | lower step p50 on each matched workload, TTFT at parity, zero watchdog fires | three short/32K, one three-concurrent group, two 150K requests |
| two-card separate process proxy | lower step p50 on every matched workload; TTFT at parity | same workload/repetition pattern; repeated on hardened runtime |
| spinning versus napping process | within noise; TTFT at parity | same workload pattern, independent fresh serve; placement rather than spinning explains earlier TTFT cost |
| late-completion corruption probe | old code 12/12 wrong, fixed code 12/12 exact | GPU probe at every tested stall point; drafts previously reported OK |
| heartbeat stall detection | three parked rounds instead of 64 | controlled stalled-proxy probe; startup-failure fallback also added |
| bounded respawn soak | 10,853 exact drafts through three SIGKILLs | x86 RTX 3090 GPU soak; no wrong drafts; separate from GB10 serving benchmarks |

Late proxy writes previously touched staging already reused by a later round. The fix builds headers and resync rows privately, sending headers and rows separately. Target verification protected final output, but corrupted drafts wasted speculation. Independent review reproduced the old failure and the corrected behavior. Detailed runtime evidence is in the [async entry](https://github.com/ProprietaryLegal/dsv41-async-hostflag-handoff).

### 3. Interference with the TP fabric

Conditions: 2026-10-22, synchronous expanded-KV offload on two GB10 nodes plus one-card x86 drafter, one request, same target/settings; paced bulk RDMA on both host links. Each loaded row is one probe; the unloaded row summarizes repeat observations, count not specified here.

| load per link, achieved | context | TTFT | decode step p50 |
|---|---|---|---|
| none | 32K / 150K | 14.55 / 68.88 s | 41.9–42.4 ms |
| 10 Gb/s, 9.93 achieved | 32K | 14.59 s, +0.3% | 42.8 ms |
| 35 Gb/s, 34.71 achieved | 32K | 14.91 s, +2.5% | 42.6 ms |
| 35 Gb/s, 34.67 achieved | 150K; load spans 60 s of the 69 s prefill | 70.31 s, +2.1% | not measured |

Near-saturation costs about 2% of prefill through the shared Spark PCIe uplink. Normal Engram plus drafter traffic is far below 10 Gb/s in these runs. The attempted second PCIe-function queue-pair setup failed as configured and was not pursued; no interface upgrade benefit is established.

### 4. Failure drills

| experiment and date | result | hardware / workload / repetitions |
|---|---|---|
| worker stopped at 25 s, restarted 31 s later; 2026-10-22 | completed 4,758 tokens from a 6,000-token budget, finish `stop`, no error; drafted step 39.6 ms → target-only 27.7 ms → drafted again | two GB10 nodes, one-card x86 drafter; one live request; target-only emits one token/step |
| same outage counters | 2,101 not-ready rounds, one timeout, five connect failures; breaker did not trip | same request; drafting resumed about 30 s after the worker restart began; reconnect backoff doubled up to 30 s |
| production recovery setting | reconnect backoff capped at 5 s | configuration change following the outage; not a new latency trial |
| proxy SIGKILL; 2026-10-23 | detected about 1 s after signal; replacement started about 1 s after detection; 3/3 requests completed | two GB10 nodes, two-card worker; three concurrent 8K generations |
| proxy SIGSTOP; 2026-10-23 | heartbeat 531 ms old versus 400 ms limit; replacement started about 1 s after detection; 2/2 completed | two GB10 nodes, two-card worker; two concurrent generations; no stopped/zombie proxy left |
| idle/resume; 2026-10-23 | ten idle minutes, no false loss, next request completed | same test serve; one idle interval |
| worker rollback rehearsal; 2026-10-23 | two-card to one-card switch 12.2 s | one host-side rehearsal; target service retained |

The worker kept its process across proxy losses and accepted a new session. Fallback means target-only generation; it is not automatic restoration of a local drafter in capacity mode. GPU window ownership fixes subsequently passed independent review, but must be included in any promotion package.

### 5. Exactness and nondeterminism

| check / statistic | observation | conditions and repeats |
|---|---|---|
| stock against itself | greedy outputs diverge at token 34 at 8K and token 4 at 32K; prompt top-5 logprob deltas up to 11.9 | 2026-10-22, two GB10 nodes; greedy, thinking off, fresh cache; paired baseline repeats |
| Engram lookup boundary | 839,040 entries per rank, zero bad rows | both layers/ranks, TCP and RDMA, both GB10 nodes versus own disk; count reconciled 2026-10-23 |
| drafter parity in live shadow mode | per-position 0.9905 / 0.973 / 0.9554 / 0.952 / 0.9338; full-block 0.9088 | 2026-10-22; 1,480 request comparisons across the shadow run; context/concurrency not stratified in this summary |
| direct timing | per-step p50 from SSE gaps | interleaved same-server worker A/Bs; context and concurrency separated |

Use lookup-boundary bytes, in-module identity, matched-prefix drafter replay, cached-token behavior and acceptance checks together. Compare generated text against the stock-versus-stock floor. Teacher-forced native reference and target-logprob replay now exist for the drafter; full target-distribution fixed-prefix comparisons remain separate work.

The diagnostic `1000 × (1 + accepted/round) / tok/s` assumes compatible counters and one bonus token per round. EOS, adaptive depth and mixed scheduling can invalidate that interpretation. It does not replace the direct timer. Matched serving tests use the same prompts, `T = 1`, `top_p = 1`, high thinking effort, priority 10, per-request salts and 1,500-token budgets, with 256K/300K finalists and four concurrent 300K requests. Prompt material and raw rows are not distributed.

### 6. What did not help on its own

Conditions: the same hardware and historical measurements above, 2026-10-22/2026-10-23. Each row retains its own scope; these results are not additive.

| idea | measured result | scope |
|---|---|---|
| CPU pinning / IRQ isolation as a cure | no demonstrated one-stream gain; observed runqueue budget about 0.010 ms/step | passive host evidence and own-thread loopback pinning, not a system-wide IRQ intervention |
| shorter transport spin / event wait | about +0.068 ms/round, saves about one CPU core | isolated x86 NIC-loopback synthetic cadence; no physical Spark cable |
| remote drafter without using freed memory | about +7–9% normalized time/round at one stream | synchronous capacity-mode suite; not the later async configuration |
| remote Engram for decode | level within noise; gain is prefill | baseline/Engram repeated suite |
| eager module loading for async safety | insufficient; explicit Triton loads still block | GB10 deadlock diagnosis |
| eager module loading or shorter interpreter interval for TTFT | neither removed the in-process cost | separate control runs |
| two-card drafter alone, synchronous | small step-time reductions in both synchronous comparison sets | same target/hardware; three short/32K, one three-concurrent group, two 150K requests |

### 7. NVLink peer memory and capture streams

With Torch expandable segments enabled, the old peer-buffer path produced illegal addresses on both drafter cards. The same code passed with expandable segments disabled. Explicit `cudaMalloc` storage outside the Torch allocator and outside graph capture fixed the peer-memory hazard.

| check, 2026-10-23 | result | scope |
|---|---|---|
| fixed peer-state lifecycle | two create/use/destroy cycles passed, zero allocator bytes left | physical RTX 3090 NVLink pair, expandable segments enabled |
| owned capture stream | explicit stream at every capture site | PyTorch 2.10 cross-device default-stream hazard; pair regression passed |

See the [two-card worker](https://github.com/ProprietaryLegal/dsv41-dspark-drafter-tp2-nvlink) for the implementation and rollback.

### 8. Exact target-greedy acceptance ceiling

Conditions: replay reported 2026-10-23; 529 rounds, four requests, five drafts, T=1; two GB10 nodes, EXL3 2.75 bpw target. The baseline drafter was captured on the Sparks. Concurrency was not stratified. KL uses returned support plus a residual bucket and available reference dumps; the coverage count is not reported.

| statistic | value |
|---|---|
| oracle / captured baseline lower-bound accepted drafts per round | 3.151 / 1.831 |
| KL(target ‖ native reference), draft positions 1–5, nats | 0.6505 / 1.4726 / 1.9825 / 2.8272 / 3.3583 |

Exact probabilities do not eliminate finite-sample uncertainty. The captured drafter score is a lower bound, so the reported gap is not a promise of recoverable acceptance. Later positions are the main adaptation target, but future target states remain unavailable to a real drafter. Full per-request results, residual mass, logit-lens bounds and bootstrap limitations are in the [method entry](https://huggingface.co/ProprietaryLegal/dsv41-exact-acceptance-ceiling).

### 9. Open qualification

KV page packing and larger context remain in progress. Offline allocator arithmetic and worker-level high-position parity are not end-to-end GB10 correctness or performance results; no result is published here for these candidates. Prefix storage has CPU/network prototype evidence, with real restore/rank synchronization and engine integration still open.

Independent review found that a waiting request unable to obtain KV could prevent useful admitted work indefinitely. The prefill-ordering fix is implemented and CPU-tested; fresh independent review and GB10 validation are pending. Earlier favorable simulation does not establish a serving improvement.

The native worker is merged opt-in only and is not serving. Its corrected attention/head candidate still fails the strict stage-zero attention gate. Earlier native timings predate that correction and cannot certify the corrected arm. Self-distillation tooling and further upstream adoption also remain in progress without a qualified serving gain.

## Upstream claims

HeNryous reports a 75K prefix restored after restart in 1.3–2.1 s versus 33.8 s. That is upstream's result, not our prefix-storage measurement. Its patch set also supplies KV packing and target-side optimization candidates; their upstream speed claims are not mixed into our tables. See the [image fork](https://github.com/ProprietaryLegal/deepseek-v41-1017-dgx-spark).

DeepSeek describes model support for the larger context window; worker support and CPU allocator arithmetic do not establish end-to-end target correctness at that limit. [DeepSeek model](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017).

## Deploy

This findings repository installs no engine overlay. For a serving deployment, follow the [overview](https://github.com/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload) and individual component instructions.

For local analysis, use the [complete exact-ceiling repository](https://github.com/ProprietaryLegal/dsv41-exact-acceptance-ceiling), which includes the capture reader and lens tool absent from this smaller copy. The included exact scorer is an archival source/aggregate snapshot; its data paths and external inputs must be supplied explicitly.

The loopback tools also retain deployment-specific discovery and address assumptions. Before running them, adapt their local address, worker discovery, paths and isolated listener allocation for your own environment. Obtain `ofl.h` and the transport library from [ofl](https://github.com/ProprietaryLegal/ofl-rdma-transport). No live target or worker request is required for the isolated loopback. Do not use loopback absolute latency as a physical link measurement.

## Rollback

Offline analysis and isolated tools do not change the engine. Stop only your own experiment and remove its generated files if needed. To roll back a deployed optimization, use that component's persistent configuration procedure: async off selects synchronous drafting; full drafter rollback restores the smaller local-drafter KV budget; Engram off selects disk lookup.

## Limits and known issues

This is a bounded hardware/workload study. Sample sizes and measurement scopes vary. No prompts, captures, tokens, calibration data, weights or raw benchmark rows are included. Aggregate analysis cannot reproduce private trajectories. Histogram upper edges, direct SSE timings, normalized diagnostics, CPU models and physical-link tests have different meanings.

The included round-trip README and helper scripts contain stale standalone path/discovery assumptions; use this page's scope and the complete component procedures. The exact scorer's capture reader and native data are intentionally external to this findings copy. Larger context, prefix restore, native fidelity and scheduler changes remain gated. No adapted drafter improvement is claimed.

## Repository layout

| path | purpose |
|---|---|
| [code/roundtrip/ofl_loopback.cpp](code/roundtrip/ofl_loopback.cpp) | isolated CPU NIC-loopback copy/poll experiments, synthetic payloads |
| [code/roundtrip/observe_x86host.sh](code/roundtrip/observe_x86host.sh) | passive host/worker contention sampler; site assumptions require adaptation |
| [code/roundtrip/run_x86host.sh](code/roundtrip/run_x86host.sh) | bounded loopback experiment driver |
| [code/roundtrip/README.md](code/roundtrip/README.md) | historical tool notes; path and discovery caveats apply |
| [code/dspark/d1/exact_ceiling.py](code/dspark/d1/exact_ceiling.py) | target-logprob replay builder/scorer snapshot; use full method repo for dependencies |
| [code/dspark/d1/results/exact_ceiling.json](code/dspark/d1/results/exact_ceiling.json) | aggregate acceptance/KL results only |
| [NOTICE](NOTICE) | attribution and licence boundary |
| [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt) | complete licence for the shipped PLI Labs tools |

## Tests

No CPU unit suite is bundled in this findings copy. Validate the aggregate with `python3 -m json.tool code/dspark/d1/results/exact_ceiling.json`; check shell syntax with `bash -n code/roundtrip/observe_x86host.sh` and `bash -n code/roundtrip/run_x86host.sh`. These checks do not run a listener or contact the target. The missing transport header/library prevents a self-contained loopback build. The full method repository supplies CPU CLI import checks.

Recorded earlier publication-copy counts belong to sibling packages: benchmark 20 passed; later Spark 221 passed / 2 skipped and worker 64 passed / 40 skipped. They are not tests of this findings package. Use the siblings' documented CPU commands and do not infer GB10 correctness from CPU success.

## Credits and licences

PLI Labs integration and analysis code is Apache-2.0. Credits: DeepSeek for the target/drafter; HeNryous for the Spark image and patch set; sudoingX for the two-node recipe; turboderp for [exllamav3](https://github.com/turboderp-org/exllamav3); [vLLM](https://github.com/vllm-project/vllm) for serving. Their software, models and licence files remain independent. See [NOTICE](NOTICE).

## Related

- `dsv41-async-hostflag-handoff`: [GitHub](https://github.com/ProprietaryLegal/dsv41-async-hostflag-handoff) · [Hugging Face](https://huggingface.co/ProprietaryLegal/dsv41-async-hostflag-handoff).
- `dsv41-remote-dspark-drafter-3090`: [GitHub](https://github.com/ProprietaryLegal/dsv41-remote-dspark-drafter-3090) · [Hugging Face](https://huggingface.co/ProprietaryLegal/dsv41-remote-dspark-drafter-3090).
- `dsv41-dspark-drafter-tp2-nvlink`: [GitHub](https://github.com/ProprietaryLegal/dsv41-dspark-drafter-tp2-nvlink) · [Hugging Face](https://huggingface.co/ProprietaryLegal/dsv41-dspark-drafter-tp2-nvlink).
- `dsv41-remote-engram-rdma`: [GitHub](https://github.com/ProprietaryLegal/dsv41-remote-engram-rdma) · [Hugging Face](https://huggingface.co/ProprietaryLegal/dsv41-remote-engram-rdma).
- `ofl-rdma-transport`: [GitHub](https://github.com/ProprietaryLegal/ofl-rdma-transport) · [Hugging Face](https://huggingface.co/ProprietaryLegal/ofl-rdma-transport).
- `dsv41-native-1017-drafter-sm86`: [GitHub](https://github.com/ProprietaryLegal/dsv41-native-1017-drafter-sm86) · [Hugging Face](https://huggingface.co/ProprietaryLegal/dsv41-native-1017-drafter-sm86).
- `dsv41-longctx-bench-tools`: [GitHub](https://github.com/ProprietaryLegal/dsv41-longctx-bench-tools) · [Hugging Face](https://huggingface.co/ProprietaryLegal/dsv41-longctx-bench-tools).
- `dsv41-exact-acceptance-ceiling`: [GitHub](https://github.com/ProprietaryLegal/dsv41-exact-acceptance-ceiling) · [Hugging Face](https://huggingface.co/ProprietaryLegal/dsv41-exact-acceptance-ceiling).
- `DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload`: [GitHub](https://github.com/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload) · [Hugging Face](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload).
- `dsv41-spark-offload-findings`: [GitHub](https://github.com/ProprietaryLegal/dsv41-spark-offload-findings) · [Hugging Face](https://huggingface.co/ProprietaryLegal/dsv41-spark-offload-findings).

Upstream forks: [Spark image and offload integration](https://github.com/ProprietaryLegal/deepseek-v41-1017-dgx-spark) · [two-node launcher](https://github.com/ProprietaryLegal/dgx-spark-deepseek-v41). [HF collection](https://huggingface.co/collections/ProprietaryLegal/deepseek-v41-flash-on-2x-dgx-spark-x86-offload-6adab604a517b426ded3916e) · [hub snapshot](https://github.com/ProprietaryLegal/pli-labs/tree/main/research/dgx-spark-offload).

<!-- pli:footer:start -->

---

## About PLI Labs

“Your proprietary work product is your proprietary intelligence.” PLI Labs is the open research arm of Proprietary Legal Intelligence. We publish legal model prunes, serving recipes for owned hardware, and open-source tools so readers can inspect our methods and test our claims before hiring us.

Founded by a practicing trial lawyer, PLI builds AI systems around a legal organization's own work product: retrieval, staged workflows, validation, and attorney review. The approach grew from systems running on live matters in a litigation practice. Our consulting work supports deployment through approved frontier APIs, private cloud, or open-weight models on hardware you own. Explore the research or contact us about your organization's work.

[Website](https://proprietarylegal.ai) · [PLI Labs](https://proprietarylegal.ai/labs.html) · [Research](https://proprietarylegal.ai/research/index.html) · [GitHub](https://github.com/ProprietaryLegal) · [Hugging Face](https://huggingface.co/ProprietaryLegal) · [Contact](https://proprietarylegal.ai/contact.html)

Proprietary Legal Intelligence LLC is a technology consultancy, not a law firm, and does not provide legal services or legal advice.

<!-- pli:footer:end -->
