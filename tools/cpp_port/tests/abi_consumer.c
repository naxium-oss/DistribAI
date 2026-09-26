/* abi_consumer.c - Feature 4 proof: a PURE C program dlopens libdai_core.so
 * and drives a multi-model training job through the stable ABI. No C++
 * symbols appear in this file; it fails to link if the ABI leaks C++ types.
 * Build: cc tools/cpp_port/tests/abi_consumer.c -o /tmp/abi_consumer -ldl
 * Run:   /tmp/abi_consumer build/cpp_port/libdai_core.so
 */
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

#include "dai_abi.h"

int main(int argc, char** argv) {
  const char* lib = argc > 1 ? argv[1] : "build/cpp_port/libdai_core.so";
  void* h = dlopen(lib, RTLD_NOW);
  if (!h) { fprintf(stderr, "dlopen failed: %s\n", dlerror()); return 1; }

  /* resolve every symbol through dlsym - the ABI surface is exactly this */
  uint32_t (*abi_version)(void) = dlsym(h, "dai_abi_version");
  dai_job* (*job_new)(void) = dlsym(h, "dai_job_new");
  dai_status (*job_add)(dai_job*, const char*, uint64_t, uint32_t) = dlsym(h, "dai_job_add_model");
  dai_status (*job_limits)(dai_job*, uint32_t, uint32_t, uint32_t) = dlsym(h, "dai_job_set_limits");
  dai_status (*job_run)(dai_job*, dai_result**, char*, size_t) = dlsym(h, "dai_job_run");
  uint32_t (*result_count)(const dai_result*) = dlsym(h, "dai_result_count");
  int (*result_ok)(const dai_result*, uint32_t) = dlsym(h, "dai_result_ok");
  const char* (*result_name)(const dai_result*, uint32_t) = dlsym(h, "dai_result_model_name");
  double (*result_loss)(const dai_result*, uint32_t) = dlsym(h, "dai_result_final_loss");
  double (*result_sps)(const dai_result*, uint32_t) = dlsym(h, "dai_result_steps_per_s");
  const double* (*result_agg)(const dai_result*, const char*, size_t*) = dlsym(h, "dai_result_aggregate");
  const uint8_t* (*result_env)(const dai_result*, uint32_t, size_t*) = dlsym(h, "dai_result_envelope");
  dai_status (*env_verify)(const uint8_t*, size_t) = dlsym(h, "dai_envelope_verify");
  dai_status (*env_loss)(const uint8_t*, size_t, double*, int*) = dlsym(h, "dai_envelope_get_loss");
  void (*result_free)(dai_result*) = dlsym(h, "dai_result_free");
  void (*job_free)(dai_job*) = dlsym(h, "dai_job_free");

  if (!abi_version || !job_new || !job_add || !job_limits || !job_run || !result_count ||
      !result_ok || !result_name || !result_loss || !result_sps || !result_agg ||
      !result_env || !env_verify || !env_loss || !result_free || !job_free) {
    fprintf(stderr, "missing ABI symbol: %s\n", dlerror());
    return 1;
  }
  if (abi_version() < DAI_ABI_VERSION) {
    fprintf(stderr, "library ABI %u older than header %u\n", abi_version(), DAI_ABI_VERSION);
    return 1;
  }
  printf("dai ABI version %u OK\n", abi_version());

  /* invalid-argument contract */
  dai_job* job = job_new();
  if (job_add(job, "", 1, 10) != DAI_ERR_INVALID) { fprintf(stderr, "empty name must reject\n"); return 1; }
  if (job_add(job, "x", 1, 0) != DAI_ERR_INVALID) { fprintf(stderr, "zero steps must reject\n"); return 1; }
  if (job_limits(job, 4, 60, 2) != DAI_ERR_INVALID) { fprintf(stderr, "mem below floor must reject\n"); return 1; }

  /* real job: three models, two concurrent sandboxes */
  job_add(job, "alpha", 42, 150);
  job_add(job, "beta", 43, 150);
  job_add(job, "gamma", 44, 150);
  job_limits(job, 64, 120, 2);

  char err[256] = {0};
  dai_result* res = NULL;
  const dai_status st = job_run(job, &res, err, sizeof(err));
  if (st != DAI_OK) { fprintf(stderr, "job_run: %d (%s)\n", st, err); return 1; }

  const uint32_t n = result_count(res);
  printf("models: %u\n", n);
  if (n != 3) { fprintf(stderr, "expected 3 results\n"); return 1; }
  int ok_all = 1;
  for (uint32_t i = 0; i < n; ++i) {
    ok_all &= result_ok(res, i) == 1;
    printf("  %s: ok=%d loss=%.6f steps_per_s=%.1f\n", result_name(res, i),
           result_ok(res, i), result_loss(res, i), result_sps(res, i));
  }
  if (!ok_all) { fprintf(stderr, "a model failed\n"); return 1; }

  /* aggregate + envelope roundtrip through the ABI */
  size_t agg_len = 0;
  const double* agg = result_agg(res, "trimmed_mean", &agg_len);
  if (!agg || agg_len != 1291) { fprintf(stderr, "aggregate missing/wrong length\n"); return 1; }
  double agg_sum = 0;
  for (size_t i = 0; i < agg_len; ++i) agg_sum += agg[i];
  printf("trimmed_mean grad_sum=%.6f (len %zu)\n", agg_sum, agg_len);

  size_t env_len = 0;
  const uint8_t* env_bytes = result_env(res, 0, &env_len);
  if (!env_bytes || env_len == 0) { fprintf(stderr, "envelope missing\n"); return 1; }
  if (env_verify(env_bytes, env_len) != DAI_OK) { fprintf(stderr, "envelope verify failed\n"); return 1; }
  double loss = 0; int have = 0;
  if (env_loss(env_bytes, env_len, &loss, &have) != DAI_OK || !have ||
      loss != result_loss(res, 0)) {
    fprintf(stderr, "envelope loss mismatch vs result accessor\n");
    return 1;
  }
  printf("envelope verify + loss extraction OK\n");

  result_free(res);
  job_free(job);
  dlclose(h);
  printf("ABI CONSUMER TEST PASSED\n");
  return 0;
}
