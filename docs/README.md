# Documentation index

<p style="max-width:42rem;margin:0 auto 1rem;line-height:1.55;text-align:center;">
  <a href="../README.md"><strong>README</strong></a> &nbsp;|&nbsp; <span>(main project README — start here)</span><br/>
  <a href="../TODO.md"><strong>TODO</strong></a> &nbsp;|&nbsp; <span>(backlog)</span><br/>
  <a href="../AGENTS.md"><strong>AGENTS</strong></a> &nbsp;|&nbsp; <span>(contributor rules)</span><br/>
  <a href="README.md"><strong>Documentation index</strong></a> &nbsp;|&nbsp; <span>(this page — docs catalog only)</span><br/>
  <a href="runbooks/deployment.md"><strong>Deploy runbook</strong></a> &nbsp;|&nbsp; <span>(local smoke + ops)</span>
</p>

---

Index of first-party Markdown under **`docs/`**. The **main** project README is at the [repository root](../README.md). Use this page only to reach API contracts, architecture notes, how-to guides, and ops runbooks.

<style>
/* GitHub may strip; harmless if dropped */
.docnav { width: 100%; border-collapse: collapse; margin: 0.5rem 0 1rem; font-size: 0.95rem; }
.docnav th, .docnav td { border: 1px solid #30363d; padding: 0.35rem 0.5rem; vertical-align: top; }
.docnav th { text-align: left; background: #161b22; }
</style>

<table class="docnav">
<thead><tr><th>Area</th><th>Documents</th></tr></thead>
<tbody>
<tr><td><strong>API</strong></td><td><a href="api/README.md">API overview</a>, <a href="api/endpoints.md">HTTP endpoints</a>, <a href="api/grpc.md">gRPC protocol</a></td></tr>
<tr><td><strong>Architecture</strong></td><td><a href="architecture/system-overview.md">System overview</a> — includes model families (<code>decoder_transformer</code>, <code>gru</code>, <code>gated_conv</code>, <code>moe_decoder</code>, <code>lstm</code>, <code>resnet_lm</code>, <code>hybrid_attn_rnn</code>, <code>dense_ffn</code>)</td></tr>
<tr><td><strong>Guides</strong></td><td>
  <a href="guides/five-minute-onboarding.md">Five-minute onboarding</a>,
  <a href="guides/production-workflow.md">Production workflow</a>,
  <a href="guides/server-operator-guide.md">Server operator</a>,
  <a href="guides/node-user-guide.md">Node user</a>,
  <a href="guides/mytrainer-submodule.md">MyTrainer submodule</a>,
  <a href="guides/packaging.md">Packaging</a>,
  <a href="guides/github-releases-setup.md">GitHub releases</a>,
  <a href="guides/beta-worker-rollout.md">Beta worker rollout</a>,
  <a href="guides/contributor-join-kit.md">Contributor join kit</a>,
  <a href="guides/ephemeral-compute-colab-kaggle.md">Colab / Kaggle workers</a>,
  <a href="guides/environment-reference.md">Environment reference</a>,
  <a href="guides/sandbox-backends.md">Sandbox backends</a>,
  <a href="guides/tls-and-mtls.md">TLS and mTLS</a>
</td></tr>
<tr><td><strong>Runbooks</strong></td><td>
  <a href="runbooks/deployment.md">Deployment</a>,
  <a href="runbooks/kubernetes.md">Kubernetes / Helm</a>,
  <a href="runbooks/ports.md">Ports</a>,
  <a href="runbooks/monitoring.md">Monitoring</a>,
  <a href="runbooks/incident-response.md">Incidents</a>,
  <a href="runbooks/beta-preprod-checklist.md">Beta pre-prod checklist</a>,
  <a href="runbooks/production-release-checklist.md">Production release checklist</a>,
  <a href="runbooks/operator-join-checklist.md">Operator join checklist</a>,
  <a href="runbooks/troubleshooting.md">Troubleshooting</a>,
  <a href="runbooks/chaos-engineering.md">Chaos</a>
</td></tr>
<tr><td><strong>Review</strong></td><td><a href="review/pr-guide-security-may-2026.md">Security patches PR guide (May 2026)</a></td></tr>
<tr><td><strong>RFCs</strong></td><td>
  <code>docs/rfcs/001</code> … <code>010</code> — design history; verify claims against <code>README.md</code> and code before treating them as current spec.
</td></tr>
<tr><td><strong>Audit</strong></td><td><a href="audit/2026-04-26-incomplete-inventory.md">2026-04-26 inventory</a>, <a href="audit/2026-05-20-backlog-closure-audit.md">2026-05-20 backlog closure</a></td></tr>
<tr><td><strong>Assets</strong></td><td><a href="assets/README.md">docs/assets/</a> — non-code artifacts (images, score dumps)</td></tr>
</tbody>
</table>

Ports and hostnames shown here are **examples**. Confirm against `services_python/orchestrator_grpc.py` and your live `.env`.

**Related root files:** [`README.md`](../README.md) (main), [`AGENTS.md`](../AGENTS.md), [`TODO.md`](../TODO.md). Ops entry: [`runbooks/deployment.md`](runbooks/deployment.md).
