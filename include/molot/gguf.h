#pragma once
// Minimal GGUF v1/v2/v3 reader (mmap, zero-copy, no dependencies).
// Covers: header, metadata KV (all scalar/string/array types), tensor table,
// data section with alignment. Dequantization lives in repack (converter),
// not here — this file only maps bytes.
#include <cstdint>
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

namespace molot::gguf {

// ggml type ids (subset sized here; unknown ids throw with the id attached).
enum DType : uint32_t {
  F32 = 0,
  F16 = 1,
  Q4_0 = 2,
  Q4_1 = 3,
  Q5_0 = 6,
  Q5_1 = 7,
  Q8_0 = 8,
  Q8_1 = 9,
  Q2_K = 10,
  Q3_K = 11,
  Q4_K = 12,
  Q5_K = 13,
  Q6_K = 14,
  Q8_K = 15,
  I8 = 24,
  I16 = 25,
  I32 = 26,
  I64 = 27,
  F64 = 28,
  BF16 = 30,
};

struct TypeInfo {
  const char* name;
  uint64_t block_elems;  // elements per quantum
  uint64_t block_bytes;  // bytes per quantum
};

inline TypeInfo type_info(uint32_t dt) {
  switch (dt) {
    case F32: return {"F32", 1, 4};
    case F16: return {"F16", 1, 2};
    case BF16: return {"BF16", 1, 2};
    case I8: return {"I8", 1, 1};
    case I16: return {"I16", 1, 2};
    case I32: return {"I32", 1, 4};
    case I64: return {"I64", 1, 8};
    case F64: return {"F64", 1, 8};
    case Q4_0: return {"Q4_0", 32, 2 + 16};
    case Q4_1: return {"Q4_1", 32, 2 + 2 + 16};
    case Q5_0: return {"Q5_0", 32, 2 + 4 + 16};
    case Q5_1: return {"Q5_1", 32, 2 + 2 + 4 + 16};
    case Q8_0: return {"Q8_0", 32, 2 + 32};
    case Q8_1: return {"Q8_1", 32, 4 + 4 + 32};
    case Q2_K: return {"Q2_K", 256, 16 + 2 + 2 + 64};
    case Q3_K: return {"Q3_K", 256, 110};
    case Q4_K: return {"Q4_K", 256, 2 + 2 + 12 + 128};
    case Q5_K: return {"Q5_K", 256, 2 + 2 + 12 + 32 + 128};
    case Q6_K: return {"Q6_K", 256, 128 + 64 + 16 + 2};
    case Q8_K: return {"Q8_K", 256, 4 + 256 + 32};
    default: throw std::runtime_error("gguf: unsupported dtype id " + std::to_string(dt));
  }
}

// metadata value types
enum MetaType : uint32_t { U8 = 0, I8M = 1, U16 = 2, I16M = 3, U32 = 4, I32M = 5, F32M = 6,
                           BOOL = 7, STR = 8, ARR = 9, U64 = 10, I64M = 11, F64M = 12 };

struct Value {
  MetaType type = U32;
  uint64_t u = 0;
  double f = 0;
  std::string s;
  uint32_t arr_type = 0;
  std::vector<Value> arr;
};

struct Tensor {
  std::string name;
  std::vector<uint64_t> dims;  // ggml order: dims[0] = columns (fastest)
  uint32_t dtype = 0;
  uint64_t offset = 0;  // relative to data section start
  uint64_t ne() const {  // element count
    uint64_t n = 1;
    for (auto d : dims) n *= d;
    return n;
  }
  uint64_t nbytes() const {
    TypeInfo ti = type_info(dtype);
    return ne() / ti.block_elems * ti.block_bytes;
  }
};

class File {
 public:
  explicit File(const char* path) {
#ifdef __linux__
    fd_ = ::open(path, O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("gguf: open failed");
    struct stat st;
    if (::fstat(fd_, &st)) throw std::runtime_error("gguf: fstat failed");
    size_ = (size_t)st.st_size;
    base_ = (const uint8_t*)::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (base_ == MAP_FAILED) throw std::runtime_error("gguf: mmap failed");
#else
    throw std::runtime_error("gguf: mmap reader is linux-only");
#endif
    parse();
  }
  ~File() {
#ifdef __linux__
    if (base_) ::munmap((void*)base_, size_);
    if (fd_ >= 0) ::close(fd_);
#endif
  }
  File(const File&) = delete;
  File& operator=(const File&) = delete;

  uint32_t version = 0;
  std::vector<std::pair<std::string, Value>> kv;  // file order
  std::vector<Tensor> tensors;
  uint64_t data_off = 0;
  uint32_t alignment = 32;

  const Value* find(const std::string& key) const {
    for (auto& kvp : kv)
      if (kvp.first == key) return &kvp.second;
    return nullptr;
  }
  // typed getters with default (missing arch keys are normal: arch-generic code
  // must tolerate absent tensors/keys — e.g. gemma4 lacks per-layer attn_k).
  uint64_t get_u64(const std::string& k, uint64_t dflt) const {
    const Value* v = find(k);
    if (!v) return dflt;
    if (v->type == U64 || v->type == U32 || v->type == U16 || v->type == U8) return v->u;
    if (v->type == I64M || v->type == I32M) return (uint64_t)(int64_t)v->u;
    throw std::runtime_error("gguf: key " + k + " is not an integer");
  }
  double get_f(const std::string& k, double dflt) const {
    const Value* v = find(k);
    if (!v) return dflt;
    if (v->type == F32M || v->type == F64M) return v->f;
    return (double)get_u64(k, 0);
  }
  std::string get_s(const std::string& k, const std::string& dflt) const {
    const Value* v = find(k);
    if (!v) return dflt;
    if (v->type != STR) throw std::runtime_error("gguf: key " + k + " is not a string");
    return v->s;
  }
  const Tensor* find_tensor(const std::string& name) const {
    for (auto& t : tensors)
      if (t.name == name) return &t;
    return nullptr;
  }
  const uint8_t* tensor_data(const Tensor& t) const { return base_ + data_off + t.offset; }

 private:
  template <typename T>
  T rd(size_t& o) const {
    T v;
    std::memcpy(&v, base_ + o, sizeof(T));
    o += sizeof(T);
    return v;
  }
  std::string rd_str(size_t& o) const {
    uint64_t n = rd<uint64_t>(o);
    std::string s((const char*)base_ + o, (size_t)n);
    o += (size_t)n;
    return s;
  }
  Value rd_val(size_t& o, uint32_t type) {
    Value v;
    v.type = (MetaType)type;
    switch (type) {
      case U8: v.u = rd<uint8_t>(o); break;
      case I8M: v.u = (uint64_t)(int64_t)rd<int8_t>(o); break;
      case U16: v.u = rd<uint16_t>(o); break;
      case I16M: v.u = (uint64_t)(int64_t)rd<int16_t>(o); break;
      case U32: v.u = rd<uint32_t>(o); break;
      case I32M: v.u = (uint64_t)(int64_t)rd<int32_t>(o); break;
      case F32M: { float f = rd<float>(o); v.f = f; break; }
      case BOOL: v.u = rd<uint8_t>(o); break;
      case STR: v.s = rd_str(o); break;
      case U64: v.u = rd<uint64_t>(o); break;
      case I64M: v.u = (uint64_t)rd<int64_t>(o); break;
      case F64M: v.f = rd<double>(o); break;
      case ARR: {
        uint32_t at = rd<uint32_t>(o);
        uint64_t n = rd<uint64_t>(o);
        v.arr_type = at;
        v.arr.reserve((size_t)n);
        if (at == STR) {
          for (uint64_t i = 0; i < n; ++i) {
            Value e;
            e.type = STR;
            e.s = rd_str(o);
            v.arr.push_back(std::move(e));
          }
        } else {
          for (uint64_t i = 0; i < n; ++i) v.arr.push_back(rd_val(o, at));
        }
        break;
      }
      default: throw std::runtime_error("gguf: unknown value type");
    }
    return v;
  }
  void parse() {
    size_t o = 0;
    uint32_t magic = rd<uint32_t>(o);
    if (magic != 0x46554747u) throw std::runtime_error("gguf: bad magic");
    version = rd<uint32_t>(o);
    if (version < 1 || version > 3) throw std::runtime_error("gguf: bad version");
    uint64_t n_tensors = version >= 2 ? rd<uint64_t>(o) : rd<uint32_t>(o);
    uint64_t n_kv = version >= 2 ? rd<uint64_t>(o) : rd<uint32_t>(o);
    for (uint64_t i = 0; i < n_kv; ++i) {
      std::string k = rd_str(o);
      uint32_t t = rd<uint32_t>(o);
      kv.emplace_back(k, rd_val(o, t));
    }
    tensors.reserve((size_t)n_tensors);
    for (uint64_t i = 0; i < n_tensors; ++i) {
      Tensor t;
      t.name = rd_str(o);
      uint32_t nd = version >= 2 ? rd<uint32_t>(o) : rd<uint32_t>(o);
      t.dims.resize(nd);
      for (uint32_t d = 0; d < nd; ++d) {
        t.dims[d] = version >= 3 ? rd<uint64_t>(o) : rd<uint32_t>(o);
      }
      t.dtype = rd<uint32_t>(o);
      t.offset = rd<uint64_t>(o);
      tensors.push_back(std::move(t));
    }
    alignment = (uint32_t)get_u64("general.alignment", 32);
    data_off = (o + alignment - 1) / alignment * alignment;
    // self-check: every tensor must land inside the file
    for (auto& t : tensors) {
      uint64_t end = 0;
      try {
        end = t.nbytes();
      } catch (...) {
        continue;
      }
      if (data_off + t.offset + end > size_)
        throw std::runtime_error("gguf: tensor " + t.name + " runs past EOF");
    }
  }

#ifdef __linux__
  int fd_ = -1;
#endif
  const uint8_t* base_ = nullptr;
  size_t size_ = 0;
};

}  // namespace molot::gguf
