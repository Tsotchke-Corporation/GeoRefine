#include <algorithm>
#include <cstdint>
#include <vector>
#include "context_exponent_packet.cuh"

namespace {
struct TrieNode {
  int32_t child[2] = {-1, -1};
  int32_t symbol = -1;
};

bool valid_shape(uint32_t k, uint32_t nc, const uint8_t *row_labels,
                 const uint8_t *col_labels, const uint32_t *codes,
                 const uint8_t *lengths, const int16_t *singletons) {
  if (!row_labels || !col_labels || !codes || !lengths || !singletons ||
      (nc != 1 && nc != 2 && nc != 4) || k < 1 || k > 32768) return false;
  for (uint32_t r = 0; r < 64; ++r) if (row_labels[r] > 255) return false;
  for (uint32_t c = 0; c < k; ++c) if (col_labels[c] >= nc) return false;
  return true;
}
}  // namespace

extern "C" int bhe1_encode_main(
    const uint8_t *exponents, uint32_t k, uint32_t nc,
    const uint8_t *row_labels, const uint8_t *col_labels,
    const uint32_t *codes, const uint8_t *lengths, const int16_t *singletons,
    uint8_t *main_out, uint32_t main_capacity, uint32_t *offsets,
    uint32_t *main_bytes, uint32_t *coding_possible) {
  if (!exponents || !main_out || !offsets || !main_bytes || !coding_possible ||
      !valid_shape(k, nc, row_labels, col_labels, codes, lengths, singletons)) return 1;
  const uint32_t streams = 64 * nc;
  const uint32_t contexts = 256 * nc;
  uint64_t bit_count = 0;
  uint64_t bit_buffer = 0;
  uint32_t buffered = 0;
  uint32_t written = 0;
  uint32_t stream = 0;
  offsets[0] = 0;
  *coding_possible = 1;

  for (uint32_t row = 0; row < 64; ++row) {
    for (uint32_t cls = 0; cls < nc; ++cls, ++stream) {
      const uint32_t context = uint32_t(row_labels[row]) * nc + cls;
      if (context >= contexts) return 1;
      bool has_code = false;
      for (uint32_t symbol = 0; symbol < 256; ++symbol)
        has_code = has_code || lengths[context * 256 + symbol] != 0;
      for (uint32_t col = 0; col < k; ++col) {
        if (col_labels[col] != cls) continue;
        const uint32_t symbol = exponents[row * k + col];
        const uint32_t slot = context * 256 + symbol;
        const uint32_t length = lengths[slot];
        if (length == 0) {
          if (singletons[context] == int16_t(symbol) && !has_code) continue;
          *coding_possible = 0;
          *main_bytes = 0;
          return 0;
        }
        if (length > 31 || (codes[slot] >> length) != 0) return 3;
        bit_buffer = (bit_buffer << length) | codes[slot];
        buffered += length;
        bit_count += length;
        while (buffered >= 8) {
          buffered -= 8;
          if (written >= main_capacity) return 4;
          main_out[written++] = uint8_t(bit_buffer >> buffered);
          bit_buffer &= buffered ? ((uint64_t(1) << buffered) - 1) : 0;
        }
      }
      if (bit_count > 0xffffffffu) return 4;
      offsets[stream + 1] = uint32_t(bit_count);
    }
  }
  if (buffered) {
    if (written >= main_capacity) return 4;
    main_out[written++] = uint8_t(bit_buffer << (8 - buffered));
  }
  *main_bytes = written;
  if (stream != streams || written != (bit_count + 7) / 8) return 4;
  return 0;
}

extern "C" int bhe1_decode_main(
    const uint8_t *main_data, uint32_t main_bytes, uint32_t bit_count,
    const uint32_t *offsets, const uint8_t *raw8, uint32_t k, uint32_t nc,
    const uint8_t *row_labels, const uint8_t *col_labels,
    const uint32_t *codes, const uint8_t *lengths, const int16_t *singletons,
    uint16_t *output) {
  if ((!main_data && main_bytes) || !offsets || !raw8 || !output ||
      !valid_shape(k, nc, row_labels, col_labels, codes, lengths, singletons)) return 1;
  if (main_bytes != (bit_count + 7) / 8 || offsets[0] != 0 || offsets[64 * nc] != bit_count) return 2;
  for (uint32_t i = 0; i < 64 * nc; ++i)
    if (offsets[i] > offsets[i + 1]) return 2;
  if ((bit_count & 7) && main_bytes &&
      (main_data[main_bytes - 1] & ((1u << (8 - (bit_count & 7))) - 1))) return 2;

  uint32_t stream = 0;
  for (uint32_t row = 0; row < 64; ++row) {
    for (uint32_t cls = 0; cls < nc; ++cls, ++stream) {
      const uint32_t context = uint32_t(row_labels[row]) * nc + cls;
      const uint32_t lo = offsets[stream], hi = offsets[stream + 1];
      uint32_t count = 0;
      for (uint32_t col = 0; col < k; ++col) count += (col_labels[col] == cls);
      bool has_code = false;
      for (uint32_t symbol = 0; symbol < 256; ++symbol)
        has_code = has_code || lengths[context * 256 + symbol] != 0;

      if (singletons[context] == -2) {
        if (count) return 5;
        if (lo != hi) return 7;
        continue;
      }
      if (singletons[context] >= 0 && !has_code) {
        if (lo != hi) return 6;
        const uint8_t exponent = uint8_t(singletons[context]);
        for (uint32_t col = 0; col < k; ++col) if (col_labels[col] == cls) {
          const uint32_t at = row * k + col;
          const uint32_t b = raw8[at];
          output[at] = uint16_t((uint32_t(exponent) << 7) | (b & 127) | ((b & 128) << 8));
        }
        continue;
      }
      if (!has_code) return 5;

      std::vector<TrieNode> trie(1);
      for (uint32_t symbol = 0; symbol < 256; ++symbol) {
        const uint32_t length = lengths[context * 256 + symbol];
        if (!length) continue;
        if (length > 31 || (codes[context * 256 + symbol] >> length) != 0) return 3;
        int32_t node = 0;
        for (int bit = int(length) - 1; bit >= 0; --bit) {
          if (trie[node].symbol >= 0) return 3;
          const uint32_t direction = (codes[context * 256 + symbol] >> bit) & 1;
          int32_t next = trie[node].child[direction];
          if (next < 0) {
            next = int32_t(trie.size());
            trie[node].child[direction] = next;
            trie.emplace_back();
          }
          node = next;
        }
        if (trie[node].symbol >= 0 || trie[node].child[0] >= 0 || trie[node].child[1] >= 0) return 3;
        trie[node].symbol = int32_t(symbol);
      }

      uint32_t cursor = lo;
      for (uint32_t col = 0; col < k; ++col) {
        if (col_labels[col] != cls) continue;
        int32_t node = 0;
        while (trie[node].symbol < 0) {
          if (cursor >= hi) return 8;
          const uint32_t bit = (main_data[cursor >> 3] >> (7 - (cursor & 7))) & 1;
          ++cursor;
          node = trie[node].child[bit];
          if (node < 0) return 9;
        }
        const uint32_t at = row * k + col;
        const uint32_t b = raw8[at];
        output[at] = uint16_t((uint32_t(trie[node].symbol) << 7) | (b & 127) | ((b & 128) << 8));
      }
      if (cursor != hi) return 7;
    }
  }
  return 0;
}

extern "C" int bhe1_split_checkpoints(
    const uint8_t* main_data,uint64_t main_bytes,const uint32_t* tables,uint64_t table_words,
    const int64_t* context_dir,const int64_t* streams,const int32_t* row_map,const int64_t* tile_rc,
    const int64_t* counts,uint32_t N,uint32_t K,uint32_t nc,uint32_t tiles,uint32_t splits,int64_t* out) {
  if(!main_data||!tables||!context_dir||!streams||!row_map||!tile_rc||!counts||!out||!N||!K||K%128||
      (nc!=1&&nc!=2&&nc!=4)||splits<1||splits>8||splits>K/128) return 1;
  const uint32_t nch=K/128; std::fill(out,out+uint64_t(N)*nc*(splits-1),0);
  for(uint32_t tile=0;tile<tiles;++tile)for(uint32_t slot=0;slot<64;++slot){
    const int32_t original_row=row_map[uint64_t(tile)*64+slot];
    if(original_row<0)continue;
    if(uint32_t(original_row)>=N)return 1;
    const uint32_t original=uint32_t(original_row);
    const uint32_t cls=uint32_t(tile_rc[tile]);
    for(uint32_t cc=0;cc<nc;++cc){
      const int64_t* span=streams+((uint64_t(tile)*64+slot)*nc+cc)*2;
      const uint64_t start=uint64_t(span[0]),end=uint64_t(span[1]);
      int64_t* dst=out+(uint64_t(original)*nc+cc)*(splits-1);
      for(uint32_t s=0;s<splits-1;++s)dst[s]=int64_t(start);
      if(start==end)continue;
      const int64_t* d=context_dir+(uint64_t(cls)*nc+cc)*4;
      if(d[0]<0||d[1]<0||uint64_t(d[0])>table_words||uint64_t(d[1])>table_words-uint64_t(d[0]))return 2;
      bhe::Context context{tables+d[0],uint32_t(d[1]),int32_t(d[2]),uint32_t(d[3])};
      bhe::Cursor cur; if(bhe::init(cur,main_data,main_bytes,start,end))return 3;
      for(uint32_t chunk=0;chunk<nch;++chunk){
        for(uint32_t s=1;s<splits;++s)if(chunk==(uint64_t(s)*nch)/splits)dst[s-1]=int64_t(start+cur.consumed);
        const int64_t* bounds=counts+uint64_t(chunk)*(nc+1);
        uint32_t produced=0,need=uint32_t(bounds[cc+1]-bounds[cc]);
        while(produced<need&&!cur.error){bhe::Group g;if(bhe::group(context,cur,need-produced,g))break;produced+=g.count;}
        if(cur.error)return 4;
      }
      if(bhe::finish(cur))return 5;
    }
  }
  return 0;
}
