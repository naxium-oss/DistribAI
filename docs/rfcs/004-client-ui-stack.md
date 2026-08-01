# RFC 004: Client UI Stack Research — Tauri vs Electron

**Status:** Historical research, not the current implementation  
**Date:** 2026-04-21  
**Research Task:** Client UI Stack Decision (Tauri vs Electron)  
**Output:** docs/rfcs/004-client-ui-stack.md

---

## Executive Summary

This RFC records earlier research comparing Tauri and Electron. It is kept for
reference, but the production desktop app now uses the Python launcher
(`grid.py`, `services_python/server_gui.py`, and
`worker/src/daemon/gui_launcher.py`) to open the bundled dashboard in a native
window and package it as Windows `.exe`/installer and macOS `.app` artifacts.
Do not treat the Tauri recommendation below as the current build plan.

---

## 1. Electron Overview

### Architecture
Electron bundles Chromium and Node.js with every application:
- **Main process**: Node.js runtime for application logic
- **Renderer process**: Chromium instances for UI rendering (one per window)
- **IPC**: Inter-process communication between main and renderer processes

### Key Characteristics
- **Bundle size**: 80-150MB (includes Chromium ~300MB + Node.js)
- **Memory usage**: 150-300MB at idle
- **Startup time**: 2-5 seconds
- **Platform support**: Windows, macOS, Linux (desktop only)
- **Security model**: Open by default, requires manual hardening
- **Development**: Pure JavaScript/TypeScript
- **Ecosystem**: Mature (10+ years), thousands of packages

### Advantages
- **Consistent rendering**: Bundled Chromium ensures identical UI across platforms
- **Mature ecosystem**: Extensive npm packages, documentation, and community support
- **JavaScript-only**: No need to learn Rust or other systems languages
- **Proven at scale**: VS Code, Slack, Discord, Figma, and many others
- **Advanced multi-window**: Native support for complex window management
- **Rapid development**: Fast iteration for JavaScript teams

### Disadvantages
- **Large bundle size**: 80-150MB installers burden users with slow connections
- **High memory usage**: 150-300MB impacts systems with limited RAM
- **Security complexity**: Requires manual hardening (context isolation, CSP, preload scripts)
- **No mobile support**: Desktop only (no iOS/Android)
- **Resource intensive**: Multiple Chromium instances for multi-window apps

---

## 2. Tauri Overview

### Architecture
Tauri uses native system WebViews with a Rust core:
- **Main process**: Rust binary (compiled to native code)
- **WebView process**: OS's native WebView (Edge WebView2 on Windows, WebKit on macOS, WebKitGTK on Linux)
- **IPC**: Message passing between WebView and Rust backend
- **No runtime**: No Node.js or Chromium bundled

### Key Characteristics
- **Bundle size**: <10MB (typically 3-8MB)
- **Memory usage**: 30-50MB at idle
- **Startup time**: <1 second
- **Platform support**: Windows, macOS, Linux, iOS, Android (desktop + mobile)
- **Security model**: Locked down by default, capability-based permissions
- **Development**: Rust backend + JavaScript/TypeScript frontend
- **Ecosystem**: Growing rapidly (2+ years), core plugins available

### Advantages
- **Tiny bundle size**: <10MB (10x smaller than Electron)
- **Low memory usage**: 30-50MB (5x lower than Electron)
- **Security by default**: Capability-based permission system, everything disabled by default
- **Mobile support**: iOS and Android from single codebase (Tauri 2.x)
- **Fast startup**: <1 second startup time
- **Native performance**: Compiled Rust binary, no runtime overhead
- **Cross-platform consistency**: Abstraction layer handles WebView differences

### Disadvantages
- **Rust learning curve**: Requires Rust knowledge for backend logic
- **Smaller ecosystem**: Fewer packages and community resources than Electron
- **WebView inconsistencies**: System WebViews may have rendering differences across platforms
- **Compilation time**: Rust compilation slower than JavaScript (20+ seconds for large apps)
- **Polyfills required**: May need CSS polyfills for cross-platform consistency
- **Fewer advanced features**: Less mature multi-window and advanced UI features

---

## 3. Comparative Analysis

| Aspect | Electron | Tauri | Winner for DistribAI |
|--------|----------|-------|---------------------|
| **Bundle Size** | 80-150MB | <10MB | Tauri (10x smaller) |
| **Memory Usage** | 150-300MB | 30-50MB | Tauri (5x lower) |
| **Startup Time** | 2-5 seconds | <1 second | Tauri (5x faster) |
| **Security** | Open by default | Locked down | Tauri (safer) |
| **Platform Support** | Desktop only | Desktop + Mobile | Tauri (future mobile) |
| **Development Speed** | Fast (JS only) | Slower (Rust + JS) | Electron (faster) |
| **Ecosystem Maturity** | Very mature | Growing | Electron (more resources) |
| **Consistency** | High (bundled) | Medium (system) | Electron (more consistent) |
| **Resource Efficiency** | Low | High | Tauri (more efficient) |

---

## 4. Suitability for DistribAI

### Use Case Analysis

DistribAI's client UI requirements:
- **Background operation**: Contributors run client 24/7
- **Resource efficiency**: Must not impact user's primary work
- **Security**: Handle authentication, encryption, and credit management
- **Cross-platform**: Windows, macOS, Linux (primary), mobile (future)
- **Installation**: Easy download and setup for contributors
- **Updates**: Seamless auto-update mechanism

### Electron Suitability

**Pros:**
- Fast development for JavaScript-focused team
- Mature auto-update ecosystem (Squirrel, electron-updater)
- Extensive documentation for common use cases
- Proven security hardening patterns available

**Cons:**
- 150MB+ burden for contributors with slow connections
- 150-300MB memory usage impacts user's system performance
- High resource usage discourages 24/7 operation
- No mobile support for future expansion
- Security requires careful implementation (context isolation, CSP, etc.)

### Tauri Suitability

**Pros:**
- <10MB download, easy for contributors to install
- 30-50MB memory usage, minimal impact on user's system
- Encourages 24/7 operation due to low resource footprint
- Security by default reduces attack surface
- Mobile support (iOS/Android) for future expansion
- Fast startup (<1 second) improves user experience
- Rust backend enables efficient system-level operations (file I/O, networking)

**Cons:**
- Requires Rust knowledge for backend logic
- Smaller ecosystem means more custom implementation
- WebView rendering may vary across platforms
- Auto-update ecosystem less mature than Electron's
- Longer compilation times during development

---

## 5. Recommendation

### Primary Recommendation: Tauri

**Rationale:**

1. **Resource Efficiency Critical**: DistribAI contributors will run the client 24/7. Tauri's 5x lower memory usage and 10x smaller bundle size significantly reduce the barrier to contribution. A 150MB Electron app is a substantial burden; a 5MB Tauri app is negligible.

2. **Security by Default**: DistribAI handles credits, authentication, and potentially cryptocurrency-related operations. Tauri's capability-based permission system (everything disabled by default) provides a stronger security foundation than Electron's open-by-default model.

3. **Future Mobile Support**: Tauri 2.x supports iOS and Android from a single codebase. While mobile is not an immediate priority, having a unified path simplifies future expansion.

4. **Rust Synergy**: DistribAI's backend is planned in Go (see crates/ directory). While Rust and Go are different, both are systems languages with similar performance characteristics. The team's comfort with systems languages makes Rust adoption more feasible.

5. **Installation Experience**: A <10MB download installs in seconds over most connections. A 150MB Electron download can take minutes on slow connections, creating friction for onboarding new contributors.

### Alternative: Electron (Fallback)

**Consider Electron if:**
- The team has no Rust expertise and cannot acquire it
- Development speed is the highest priority
- The project requires advanced multi-window UI features not available in Tauri
- WebView rendering inconsistencies across platforms are unacceptable
- Timeline is extremely tight and cannot accommodate Rust learning curve

---

## 6. Implementation Plan

### If Tauri is Chosen

#### Phase 0 (Proof of Concept - 2 weeks)
1. **Setup Tauri development environment**
   - Install Rust toolchain
   - Create Tauri app with React/Vue/Svelte frontend
   - Implement basic UI (status, start/stop, settings)

2. **Implement core worker integration**
   - WebSocket connection to orchestrator
   - Job execution status display
   - Credit tracking display

3. **Benchmark**
   - Measure bundle size, memory usage, startup time
   - Compare against mock Electron implementation

#### Phase 1 (MVP - 4 weeks)
1. **Complete UI implementation**
   - Dashboard with training progress
   - Settings panel (GPU selection, bandwidth limits)
   - Authentication flow

2. **System integration**
   - Auto-start on OS boot
   - Tray icon with status
   - File system access for model weights

3. **Auto-update mechanism**
   - Implement Tauri's updater plugin
   - Test update flow on all platforms

#### Phase 2 (Production - 2 weeks)
1. **Security hardening**
   - Audit capability permissions
   - Implement secure credential storage
   - Code signing for all platforms

2. **Performance optimization**
   - Profile memory usage
   - Optimize WebView rendering
   - Minimize bundle size

3. **Documentation**
   - Installation guide
   - Troubleshooting guide
   - Contributor onboarding

### If Electron is Chosen

#### Phase 0 (Proof of Concept - 1 week)
1. **Setup Electron development environment**
2. **Implement basic UI with React/Vue/Svelte**
3. **Implement worker integration**

#### Phase 1 (MVP - 3 weeks)
1. **Complete UI implementation**
2. **System integration**
3. **Auto-update with electron-updater**

#### Phase 2 (Production - 2 weeks)
1. **Security hardening** (context isolation, CSP, preload scripts)
2. **Performance optimization**
3. **Documentation**

---

## 7. Team Considerations

### Skill Requirements

**Tauri:**
- **Must have**: 1-2 team members with Rust experience or willingness to learn
- **Nice to have**: Experience with WebView rendering, cross-platform development
- **Learning curve**: 2-4 weeks for JavaScript developers to become productive in Rust

**Electron:**
- **Must have**: JavaScript/TypeScript expertise (already present)
- **Nice to have**: Experience with Node.js, Chromium internals
- **Learning curve**: Minimal for existing web developers

### Development Speed

**Tauri:**
- Initial setup: 2-3 days
- Feature development: 20-30% slower due to Rust compilation
- Debugging: More complex (Rust + JavaScript)
- Overall: 20-30% slower than Electron

**Electron:**
- Initial setup: 1 day
- Feature development: Fast (JavaScript only)
- Debugging: Straightforward (DevTools)
- Overall: Fastest option

---

## 8. Cost-Benefit Analysis

### Tauri

**Costs:**
- 2-4 weeks Rust learning curve
- 20-30% slower development
- Smaller ecosystem (more custom implementation)
- Potential WebView rendering inconsistencies

**Benefits:**
- 10x smaller bundle size (better UX)
- 5x lower memory usage (better for 24/7 operation)
- Security by default (reduced attack surface)
- Mobile support (future expansion)
- Faster startup (better UX)
- Rust performance (efficient system operations)

**Net Impact**: Higher upfront cost, but significant long-term benefits in user experience, security, and resource efficiency.

### Electron

**Costs:**
- 10x larger bundle size (worse UX)
- 5x higher memory usage (worse for 24/7 operation)
- Security requires manual hardening
- No mobile support (future rework required)
- Slower startup (worse UX)

**Benefits:**
- Faster development (JavaScript only)
- Mature ecosystem (more packages, documentation)
- Consistent rendering (bundled Chromium)
- Proven at scale (many successful apps)

**Net Impact**: Lower upfront cost, but long-term costs in user experience, resource usage, and future mobile expansion.

---

## 9. Decision Matrix

| Factor | Weight | Tauri Score | Electron Score | Weighted Score |
|--------|--------|-------------|----------------|----------------|
| Bundle Size | 0.20 | 10 | 2 | Tauri: 2.0, Electron: 0.4 |
| Memory Usage | 0.20 | 10 | 2 | Tauri: 2.0, Electron: 0.4 |
| Security | 0.15 | 9 | 5 | Tauri: 1.35, Electron: 0.75 |
| Mobile Support | 0.10 | 10 | 0 | Tauri: 1.0, Electron: 0.0 |
| Development Speed | 0.15 | 5 | 10 | Tauri: 0.75, Electron: 1.5 |
| Ecosystem Maturity | 0.10 | 6 | 10 | Tauri: 0.6, Electron: 1.0 |
| Rendering Consistency | 0.10 | 6 | 9 | Tauri: 0.6, Electron: 0.9 |
| **Total** | 1.00 | | | **Tauri: 8.3, Electron: 4.95** |

**Winner**: Tauri (68% higher weighted score)

---

## 10. Final Recommendation

**Choose Tauri** for the DistribAI client UI.

The resource efficiency, security model, and future mobile support of Tauri align with DistribAI's requirements for a contributor-facing application that runs 24/7. While the Rust learning curve and smaller ecosystem present challenges, the long-term benefits in user experience, security, and cross-platform capabilities outweigh the upfront costs.

**Mitigation for Rust learning curve:**
- Allocate 2-4 weeks for team training
- Start with simple Rust backend (minimal logic)
- Use Tauri's JavaScript API for most functionality
- Hire or consult with Rust expert if needed

**Mitigation for smaller ecosystem:**
- Leverage Tauri's core plugins (auto-updater, notifications, file system)
- Implement custom functionality in Rust (efficient and secure)
- Contribute back to Tauri ecosystem where possible

**Next step:** Begin Tauri proof-of-concept implementation to validate technical feasibility and team productivity.

---

## 11. Open Questions

1. **Team Rust expertise**: Does the current team have Rust experience? If not, is there budget for training or hiring?
2. **WebView rendering**: Are there specific UI requirements that may have rendering inconsistencies across system WebViews?
3. **Auto-update maturity**: Is Tauri's auto-update plugin production-ready for DistribAI's needs?
4. **Timeline**: Can the project accommodate 2-4 weeks of Rust learning curve?
5. **Mobile timeline**: When is mobile support needed? If >1 year, could Electron be used now with Tauri later?

---

## Conclusion

Tauri is the recommended choice for DistribAI's client UI due to its superior resource efficiency, security model, and mobile support. While Electron offers faster development and a more mature ecosystem, Tauri's advantages in bundle size (10x smaller), memory usage (5x lower), and security (locked down by default) are critical for a contributor-facing application that runs 24/7. The Rust learning curve is a manageable cost that pays dividends in long-term user experience and system efficiency.

**Next step:** Proceed to RFC 005 on Federated Learning Aggregation Strategies.
