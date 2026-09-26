/* dai_abi.h - stable C ABI over the C++ train core (Feature 4).
 *
 * A future worker (Python, Rust, Go, another C++ binary) can dlopen
 * libdai_core.so and drive training without importing Python or linking
 * C++ standard libraries. All handles are opaque; all strings are
 * NUL-terminated UTF-8 borrowed from the caller unless marked out.
 * Thread safety: dai_train_job is thread-safe; other calls are not
 * unless documented.
 *
 * Version contract: DAI_ABI_VERSION bump = breaking change. Functions
 * returning dai_status write a human-readable error into err buffer.
 */
#ifndef DAI_ABI_H
#define DAI_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DAI_ABI_VERSION 1u

typedef struct dai_job dai_job;          /* opaque job builder */
typedef struct dai_result dai_result;    /* opaque result set */

typedef enum dai_status {
  DAI_OK = 0,
  DAI_ERR_INVALID = 1,       /* null/invalid argument */
  DAI_ERR_PARSE = 2,         /* manifest or envelope malformed */
  DAI_ERR_SANDBOX = 3,       /* child failed / limit violation */
  DAI_ERR_CKPT = 4,          /* checkpoint IO or verification */
  DAI_ERR_VERSION = 5        /* ABI/envelope version mismatch */
} dai_status;

/* ABI version of the loaded library (>= DAI_ABI_VERSION from this header means usable). */
uint32_t dai_abi_version(void);

/* ---- job builder: register models then run ---- */

dai_job* dai_job_new(void);
/* Returns DAI_OK, or DAI_ERR_INVALID for empty name / zero steps. */
dai_status dai_job_add_model(dai_job* j, const char* name, uint64_t seed, uint32_t steps);
/* Resource limits applied to every sandboxed child (mirrors run_limited). */
dai_status dai_job_set_limits(dai_job* j, uint32_t mem_mb, uint32_t cpu_sec, uint32_t max_concurrent);
/* Run all registered models concurrently. Returns DAI_OK when every model
 * trained; partial failures still produce a result set (inspect per model). */
dai_status dai_job_run(dai_job* j, dai_result** out, char* err, size_t err_len);
void dai_result_free(dai_result* r);
void dai_job_free(dai_job* j);

/* ---- result access ---- */

uint32_t dai_result_count(const dai_result* r);
/* 1 when the model at index i trained successfully. */
int dai_result_ok(const dai_result* r, uint32_t i);
const char* dai_result_model_name(const dai_result* r, uint32_t i);
double dai_result_final_loss(const dai_result* r, uint32_t i);
double dai_result_steps_per_s(const dai_result* r, uint32_t i);
/* Gradient aggregate (trimmed mean) for the whole job. out_len receives the
 * element count (1291 for 1K models). Buffer is owned by the result set and
 * valid until dai_result_free. */
const double* dai_result_aggregate(const dai_result* r, const char* method, size_t* out_len);
/* Full envelope bytes (versioned wire format) for a model's grad_report. */
const uint8_t* dai_result_envelope(const dai_result* r, uint32_t i, size_t* out_len);

/* ---- envelope helpers (for hosts that build/inspect envelopes) ---- */

/* Verify an envelope blob; returns DAI_OK when magic/version/CRC are valid. */
dai_status dai_envelope_verify(const uint8_t* data, size_t n);
/* Extract final_loss (tag 7) as f64; ok=0 when absent. */
dai_status dai_envelope_get_loss(const uint8_t* data, size_t n, double* out, int* ok);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* DAI_ABI_H */
