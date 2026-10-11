#pragma once
// .molot pack format: a derived, mmap-ready cache of a GGUF file containing
// exactly what the runtime consumes — nothing more.
//
// Layout (all integers little-endian, x86_64 LE only):
//   header      : magic "MOLT", u32 version(=1), u64 source_hash (FNV-1a of
//                 the GGUF file), u64 source_size, HParams, layer table,
//                 u64 n_tensors, tensor entries, tokenizer blobs, chat template
//   data        : 64B-aligned blobs, one per tensor entry, in entry order
//
// Tensor dtypes (ours, not GGUF's):
//   F32=0     : float32 row-major (norms, scales, baked rope freqs)
//   Q4PACK=1  : [nibbles u8][scales f32][zeros i8][wsums i16] (see quant.h)
//   Q8PACK=2  : [quants i8][scales f32][bsums i32]
//   Q4RAW=3   : verbatim GGUF Q4_0 blocks (gather path dequants per row)
//   Q8RAW=4   : verbatim GGUF Q8_0 blocks (gather path dequant per row)
//   B16RAW=5  : verbatim BF16 (widened once at load; 55MB, ~50ms)
//
// Semantics: the pack is a CACHE. The loader trusts it (no revalidation
// beyond magic/version); provenance (hash+size) lets gguf2molot --check
// confirm it still matches its GGUF. Never the only copy of anything.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __linux__
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace molot::pack {

inline constexpr uint32_t MAGIC = 0x544C4F4Du;  // "MOLT" LE
inline constexpr uint32_t VERSION = 2;

enum DType : uint32_t { F32 = 0, Q4PACK = 1, Q8PACK = 2, Q4RAW = 3, Q8RAW = 4, B16RAW = 5 };

struct HParams {
  char arch[64] = {};
  uint32_t n_layer = 0, n_embd = 0, n_head = 0, n_kv = 0, n_ffn = 0;
  float eps = 1e-6f, rope_base = 10000, rope_swa_base = 10000, logit_cap = 0;
  uint32_t rope_dim = 0, rope_swa_dim = 0, sw_window = 0;
  uint32_t vocab = 0;
  int32_t bos = 1, eos = 2;
  uint32_t has_layer_emb = 0, layer_emb_dim = 0;
};

struct LayerRec {
  uint8_t has_kv = 0, swa = 0;
  uint8_t pad[2] = {};
  int32_t kv_src = -1;
  uint32_t qhd = 0, khd = 0, window = 0;
  float out_scale = 1.0f;
};

struct TensorRec {
  std::string name;
  uint32_t dtype = 0;
  uint64_t rows = 0, cols = 0;  // matrix shape (rows=dims[1]); 0 for vectors
  uint64_t nelems = 0;          // logical elements (rows*cols, or count)
  uint64_t nbytes = 0;
  uint64_t offset = 0;  // from data section start
};

inline uint64_t fnv1a_file(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) throw std::runtime_error("pack: cannot hash source");
  uint64_t h = 14695981039346656037ull;
  char buf[1 << 20];
  size_t n;
  while ((n = fread(buf, 1, sizeof buf, f)) > 0)
    for (size_t i = 0; i < n; ++i) {
      h ^= (uint8_t)buf[i];
      h *= 1099511628211ull;
    }
  fclose(f);
  return h;
}
inline uint64_t file_size(const char* path) {
  struct stat st;
  if (stat(path, &st)) throw std::runtime_error("pack: cannot stat source");
  return (uint64_t)st.st_size;
}

struct Pack {
  HParams hp;
  std::vector<LayerRec> layers;
  std::vector<TensorRec> tensors;
  std::vector<std::string> tok_pieces;
  std::vector<std::string> tok_merges;
  std::string chat_template;
  std::string tok_model;  // tokenizer.ggml.model ("gpt2" or other)
  // data blobs in tensor order (writer-side) / mmap base (reader-side)
  std::vector<std::vector<uint8_t>> blobs;
  uint64_t source_hash = 0, source_size = 0;
};

inline void write_u32(FILE* f, uint32_t v) { fwrite(&v, 4, 1, f); }
inline void write_u64(FILE* f, uint64_t v) { fwrite(&v, 8, 1, f); }
inline void write_bytes(FILE* f, const void* p, size_t n) { fwrite(p, 1, n, f); }
inline void write_str(FILE* f, const std::string& s) {
  write_u64(f, s.size());
  if (!s.empty()) write_bytes(f, s.data(), s.size());
}

inline void write_pack(const char* path, const Pack& p) {
  FILE* f = fopen(path, "wb");
  if (!f) throw std::runtime_error("pack: cannot open output");
  write_u32(f, MAGIC);
  write_u32(f, VERSION);
  write_u64(f, p.source_hash);
  write_u64(f, p.source_size);
  write_bytes(f, &p.hp, sizeof(HParams));
  write_u32(f, (uint32_t)p.layers.size());
  if (!p.layers.empty()) write_bytes(f, p.layers.data(), p.layers.size() * sizeof(LayerRec));
  write_u64(f, p.tensors.size());
  uint64_t off = 0;
  for (auto& t : p.tensors) {
    write_str(f, t.name);
    write_u32(f, t.dtype);
    write_u64(f, t.rows);
    write_u64(f, t.cols);
    write_u64(f, t.nelems);
    write_u64(f, t.nbytes);
    off = (off + 63) / 64 * 64;
    write_u64(f, off);
    off += t.nbytes;
  }
  write_u64(f, p.tok_pieces.size());
  for (auto& s : p.tok_pieces) write_str(f, s);
  write_u64(f, p.tok_merges.size());
  for (auto& s : p.tok_merges) write_str(f, s);
  write_str(f, p.chat_template);
  write_str(f, p.tok_model);
  if (p.blobs.size() != p.tensors.size())
    throw std::runtime_error("pack: blobs != tensors");
  // pad to 64B before data
  long pos = ftell(f);
  size_t pad = (size_t)((64 - (pos % 64)) % 64);
  static const char zeros[64] = {};
  if (pad) write_bytes(f, zeros, pad);
  for (size_t i = 0; i < p.tensors.size(); ++i) {
    long at = ftell(f);
    // offsets were precomputed sequentially; just pad to 64B
    size_t mis = (size_t)(at % 64);
    if (mis) write_bytes(f, zeros, 64 - mis);
    if (p.blobs[i].size() != p.tensors[i].nbytes)
      throw std::runtime_error("pack: blob size mismatch for " + p.tensors[i].name);
    if (!p.blobs[i].empty()) write_bytes(f, p.blobs[i].data(), p.blobs[i].size());
  }
  fclose(f);
}

struct Reader {
#ifdef __linux__
  int fd = -1;
#endif
  const uint8_t* base = nullptr;
  size_t size = 0;
  Pack meta;  // header decoded (blobs empty; use blob(i) accessor)
  uint64_t data_off = 0;

  explicit Reader(const char* path) {
#ifdef __linux__
    fd = ::open(path, O_RDONLY);
    if (fd < 0) throw std::runtime_error("pack: open failed");
    struct stat st;
    if (::fstat(fd, &st)) throw std::runtime_error("pack: fstat failed");
    size = (size_t)st.st_size;
    base = (const uint8_t*)::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) throw std::runtime_error("pack: mmap failed");
#else
    throw std::runtime_error("pack: mmap reader is linux-only");
#endif
    size_t o = 0;
    auto rd = [&](void* p, size_t n) {
      if (o + n > size) throw std::runtime_error("pack: truncated");
      std::memcpy(p, base + o, n);
      o += n;
    };
    uint32_t magic = 0, ver = 0;
    rd(&magic, 4);
    rd(&ver, 4);
    if (magic != MAGIC) throw std::runtime_error("pack: bad magic (not a .molot file?)");
    if (ver != VERSION) throw std::runtime_error("pack: version mismatch");
    rd(&meta.source_hash, 8);
    rd(&meta.source_size, 8);
    rd(&meta.hp, sizeof(HParams));
    uint32_t nl = 0;
    rd(&nl, 4);
    meta.layers.resize(nl);
    if (nl) rd(meta.layers.data(), nl * sizeof(LayerRec));
    uint64_t nt = 0;
    rd(&nt, 8);
    meta.tensors.reserve(nt);
    for (uint64_t i = 0; i < nt; ++i) {
      TensorRec t;
      uint64_t sl = 0;
      rd(&sl, 8);
      if (o + sl > size) throw std::runtime_error("pack: truncated name");
      t.name.assign((const char*)base + o, (size_t)sl);
      o += (size_t)sl;
      rd(&t.dtype, 4);
      rd(&t.rows, 8);
      rd(&t.cols, 8);
      rd(&t.nelems, 8);
      rd(&t.nbytes, 8);
      rd(&t.offset, 8);
      meta.tensors.push_back(std::move(t));
    }
    uint64_t np = 0;
    rd(&np, 8);
    for (uint64_t i = 0; i < np; ++i) {
      uint64_t sl = 0;
      rd(&sl, 8);
      if (o + sl > size) throw std::runtime_error("pack: truncated piece");
      meta.tok_pieces.emplace_back((const char*)base + o, (size_t)sl);
      o += (size_t)sl;
    }
    uint64_t nm = 0;
    rd(&nm, 8);
    for (uint64_t i = 0; i < nm; ++i) {
      uint64_t sl = 0;
      rd(&sl, 8);
      if (o + sl > size) throw std::runtime_error("pack: truncated merge");
      meta.tok_merges.emplace_back((const char*)base + o, (size_t)sl);
      o += (size_t)sl;
    }
    {
      uint64_t sl = 0;
      rd(&sl, 8);
      if (o + sl > size) throw std::runtime_error("pack: truncated template");
      meta.chat_template.assign((const char*)base + o, (size_t)sl);
      o += (size_t)sl;
    }
    {
      uint64_t sl = 0;
      rd(&sl, 8);
      if (o + sl > size) throw std::runtime_error("pack: truncated tokmodel");
      meta.tok_model.assign((const char*)base + o, (size_t)sl);
      o += (size_t)sl;
    }
    data_off = (o + 63) / 64 * 64;
    // validate blob ranges
    for (auto& t : meta.tensors)
      if (data_off + t.offset + t.nbytes > size)
        throw std::runtime_error("pack: tensor past EOF: " + t.name);
  }
  ~Reader() {
#ifdef __linux__
    if (base) ::munmap((void*)base, size);
    if (fd >= 0) ::close(fd);
#endif
  }
  Reader(const Reader&) = delete;
  Reader& operator=(const Reader&) = delete;

  const uint8_t* blob(const TensorRec& t) const { return base + data_off + t.offset; }
  const TensorRec* find(const std::string& name) const {
    for (auto& t : meta.tensors)
      if (t.name == name) return &t;
    return nullptr;
  }
};

}  // namespace molot::pack
