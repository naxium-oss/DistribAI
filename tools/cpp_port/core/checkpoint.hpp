// Checkpoint format (see runtime/baselines/design_envelope_checkpoint.md).
//
// File = envelope (kind=CheckpointRef) + binary section:
//   [32B header + TLVs declaring weights/optim/rng lengths]
//   [weights f32 blob in parameters() order][AdamW m][AdamW v][MT19937 state]
// Atomic write via tmp+rename, retention keeps newest N per model.
// RNG continuity: stores the full 624-word generator state and index so the
// resumed run draws exactly the stream the crashed run would have drawn.
#pragma once

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "envelope.hpp"
#include "mt19937.hpp"
#include "parity.hpp"

namespace distribai {

struct CheckpointMeta {
  std::string model_name;
  std::string checkpoint_id;
  uint64_t seed = 0;
  uint64_t steps = 0;      // resume point (steps completed)
  uint64_t n_params = 0;
  uint64_t weights_len = 0;
  uint64_t optim_len = 0;
  uint64_t rng_len = 0;
  uint64_t payload_len = 0;
  int64_t opt_steps = 0;  // AdamW bias-correction step counter
};

// Full RNG state capture (mt table + index) - aliases TorchRng::State.
using RngState = TorchRng::State;

class Checkpoint {
 public:
  // Capture model + optimizer + RNG. model_name must be sanitized already.
  static Checkpoint capture(const std::string& model_name, uint64_t seed, uint64_t steps_done,
                            const ParityMlp& model, const ParityAdamW& opt, const TorchRng& rng) {
    Checkpoint ck;
    ck.meta_.model_name = model_name;
    ck.meta_.seed = seed;
    ck.meta_.steps = steps_done;
    ck.meta_.n_params = static_cast<uint64_t>(model.param_count());

    const auto w = model.flat_weights();
    ck.weights_.resize(w.size() * 4);
    std::memcpy(ck.weights_.data(), w.data(), ck.weights_.size());

    // optimizer state: m then v, each in parameters() order; the AdamW step
    // counter is stored separately in the envelope (not inside the blobs).
    std::vector<float> m, v;
    opt.export_state(m, v);
    ck.optim_.resize((m.size() + v.size()) * 4);
    std::memcpy(ck.optim_.data(), m.data(), m.size() * 4);
    std::memcpy(ck.optim_.data() + m.size() * 4, v.data(), v.size() * 4);
    ck.opt_steps_ = opt.export_steps();

    RngState rs = rng.export_state();
    ck.rng_.resize(sizeof(rs));
    std::memcpy(ck.rng_.data(), &rs, sizeof(rs));

    ck.meta_.weights_len = ck.weights_.size();
    ck.meta_.optim_len = ck.optim_.size();
    ck.meta_.rng_len = ck.rng_.size();
    ck.meta_.payload_len = ck.weights_.size() + ck.optim_.size() + ck.rng_.size();
    ck.opt_steps_ = opt.export_steps();
    return ck;
  }

  const CheckpointMeta& meta() const { return meta_; }
  const std::vector<uint8_t>& weights() const { return weights_; }
  const std::vector<uint8_t>& optim() const { return optim_; }
  const std::vector<uint8_t>& rng() const { return rng_; }

  // Atomic write to <dir>/<id>.ckp. Returns the final path.
  std::string save(const std::string& dir, uint64_t keep = 3) const {
    ::mkdir(dir.c_str(), 0755);
    const std::string id = checkpoint_id();
    const std::string final_path = dir + "/" + id + ".ckp";
    const std::string tmp_path = dir + "/." + id + ".tmp-" + std::to_string(::getpid());

    env::Envelope e(env::Kind::CheckpointRef, 0);
    e.add_str(env::TAG_MODEL_NAME, meta_.model_name);
    e.add_str(env::TAG_CHECKPOINT_ID, id);
    e.add_u64(env::TAG_SEED, meta_.seed);
    e.add_u64(env::TAG_STEPS, meta_.steps);
    e.add_u64(env::TAG_N_PARAMS, meta_.n_params);
    e.add_u64(env::TAG_CKPT_WEIGHTS_LEN, meta_.weights_len);
    e.add_u64(env::TAG_CKPT_OPTIM_LEN, meta_.optim_len);
    e.add_u64(env::TAG_CKPT_RNG_LEN, meta_.rng_len);
    e.add_u64(env::TAG_CKPT_PAYLOAD_LEN, meta_.payload_len);
    e.add_u64(env::TAG_OPTIM_STEPS, static_cast<uint64_t>(opt_steps_));

    std::vector<uint8_t> bytes = e.encode();
    bytes.insert(bytes.end(), weights_.begin(), weights_.end());
    bytes.insert(bytes.end(), optim_.begin(), optim_.end());
    bytes.insert(bytes.end(), rng_.begin(), rng_.end());

    FILE* f = std::fopen(tmp_path.c_str(), "wb");
    if (!f) throw std::runtime_error("checkpoint tmp open failed: " + tmp_path);
    const size_t w1 = std::fwrite(bytes.data(), 1, bytes.size(), f);
    std::fflush(f);
    ::fsync(::fileno(f));
    std::fclose(f);
    if (w1 != bytes.size()) {
      ::unlink(tmp_path.c_str());
      throw std::runtime_error("checkpoint short write");
    }
    if (::rename(tmp_path.c_str(), final_path.c_str()) != 0) {
      ::unlink(tmp_path.c_str());
      throw std::runtime_error("checkpoint rename failed");
    }
    retain(dir, meta_.model_name, keep);
    return final_path;
  }

  // Load + verify envelope CRC and declared section lengths.
  static bool load(const std::string& path, Checkpoint& out, std::string& err) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open " + path; return false; }
    std::vector<uint8_t> bytes;
    uint8_t buf[65536];
    size_t r;
    while ((r = std::fread(buf, 1, sizeof(buf), f)) > 0) bytes.insert(bytes.end(), buf, buf + r);
    std::fclose(f);

    env::Envelope e;
    if (!env::Envelope::decode(bytes.data(), bytes.size(), e, err)) return false;
    if (e.kind() != env::Kind::CheckpointRef) { err = "not a checkpoint envelope"; return false; }

    Checkpoint ck;
    std::string s;
    if (!e.get_str(env::TAG_MODEL_NAME, s)) { err = "missing model_name"; return false; }
    ck.meta_.model_name = s;
    if (!e.get_str(env::TAG_CHECKPOINT_ID, s)) { err = "missing checkpoint_id"; return false; }
    ck.meta_.checkpoint_id = s;
    uint64_t u;
    if (e.get_u64(env::TAG_SEED, u)) ck.meta_.seed = u;
    if (e.get_u64(env::TAG_STEPS, u)) ck.meta_.steps = u;
    if (e.get_u64(env::TAG_N_PARAMS, u)) ck.meta_.n_params = u;
    if (e.get_u64(env::TAG_OPTIM_STEPS, u)) ck.opt_steps_ = static_cast<int64_t>(u);
    if (!e.get_u64(env::TAG_CKPT_WEIGHTS_LEN, ck.meta_.weights_len) ||
        !e.get_u64(env::TAG_CKPT_OPTIM_LEN, ck.meta_.optim_len) ||
        !e.get_u64(env::TAG_CKPT_RNG_LEN, ck.meta_.rng_len) ||
        !e.get_u64(env::TAG_CKPT_PAYLOAD_LEN, ck.meta_.payload_len)) {
      err = "missing section lengths"; return false;
    }
    const size_t head = 32;  // TODO: use actual envelope size from decode
    const size_t env_size = envelope_size(bytes);
    (void)head;
    const size_t payload_off = env_size;
    if (payload_off + ck.meta_.payload_len > bytes.size()) { err = "payload truncated"; return false; }

    ck.weights_.assign(bytes.begin() + payload_off,
                       bytes.begin() + payload_off + ck.meta_.weights_len);
    ck.optim_.assign(bytes.begin() + payload_off + ck.meta_.weights_len,
                     bytes.begin() + payload_off + ck.meta_.weights_len + ck.meta_.optim_len);
    ck.rng_.assign(bytes.begin() + payload_off + ck.meta_.weights_len + ck.meta_.optim_len,
                   bytes.begin() + payload_off + ck.meta_.payload_len);
    if (ck.weights_.size() % 4 != 0 || ck.optim_.size() % 4 != 0) { err = "misaligned blobs"; return false; }
    out = std::move(ck);
    return true;
  }

  // Restore into a live model/optimizer/rng. Returns false on shape mismatch.
  bool restore(ParityMlp& model, ParityAdamW& opt, TorchRng& rng) const {
    if (weights_.size() != static_cast<size_t>(model.param_count()) * 4) return false;
    if (optim_.size() % 2 != 0) return false;
    std::vector<float> w(model.param_count());
    std::memcpy(w.data(), weights_.data(), weights_.size());
    model.load_flat_weights(w);
    const size_t half = optim_.size() / 2;  // bytes; m and v have equal counts
    std::vector<float> m(half / 4), v(half / 4);
    std::memcpy(m.data(), optim_.data(), half);
    std::memcpy(v.data(), optim_.data() + half, half);
    opt.import_state(m, v, opt_steps_);
    if (rng_.size() != sizeof(RngState)) return false;
    RngState rs;
    std::memcpy(&rs, rng_.data(), sizeof(rs));
    rng.import_state(rs);
    return true;
  }

  std::string checkpoint_id() const {
    if (!meta_.checkpoint_id.empty()) return meta_.checkpoint_id;
    return make_id(meta_.model_name, meta_.steps);
  }

  static std::string make_id(const std::string& model, uint64_t steps) {
    char ts[32];
    std::time_t t = std::time(nullptr);
    std::tm tm{};
    localtime_r(&t, &tm);
    std::strftime(ts, sizeof(ts), "%Y%m%d-%H%M%S", &tm);
    std::string id = model + "-s" + std::to_string(steps) + "-" + ts + "-" +
                     std::to_string(::getpid());
    for (char& c : id) {
      if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '.' || c == '_')) c = '_';
    }
    return id;
  }

  // Keep the newest `keep` checkpoints for a model (by mtime), delete the rest.
  // Ordering uses NANOSECOND mtime (st_mtim): plain st_mtime has 1s granularity,
  // so bursty saves tie and the unstable sort could delete files newer than the
  // cutoff - including the checkpoint save() just returned (suite-caught bug).
  static void retain(const std::string& dir, const std::string& model, size_t keep) {
    DIR* d = ::opendir(dir.c_str());
    if (!d) return;
    std::vector<std::pair<long long, std::string>> found;
    const std::string prefix = model + "-s";
    while (dirent* de = ::readdir(d)) {
      std::string name = de->d_name;
      if (name.size() > 4 && name.rfind(".ckp") == name.size() - 4 && name.rfind(prefix, 0) == 0) {
        struct stat st{};
        if (::stat((dir + "/" + name).c_str(), &st) == 0)
          found.emplace_back(static_cast<long long>(st.st_mtim.tv_sec) * 1000000000LL +
                                 static_cast<long long>(st.st_mtim.tv_nsec),
                             name);
      }
    }
    ::closedir(d);
    if (found.size() <= keep) return;
    std::sort(found.begin(), found.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    for (size_t i = keep; i < found.size(); ++i) ::unlink((dir + "/" + found[i].second).c_str());
  }

 private:
  // Re-derive the encoded envelope size so the binary section offset is exact.
  static size_t envelope_size(const std::vector<uint8_t>& bytes) {
    if (bytes.size() < 32) return 0;
    // payload_len is at header offset 16
    uint32_t payload_len = 0;
    for (int i = 0; i < 4; ++i) payload_len |= uint32_t(bytes[16 + i]) << (8 * i);
    return 32 + payload_len;
  }

  CheckpointMeta meta_;
  int64_t opt_steps_ = 0;
  std::vector<uint8_t> weights_, optim_, rng_;
};

}  // namespace distribai
