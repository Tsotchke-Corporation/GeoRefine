"""Literal paired-vote decoder proof on synthetic BF16 words; CPU only."""
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from glc_loader import codec_v21
from glc_serve import bigemm_tbe21_ref


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "glc_serve/bi_tbe21_v2_csrc"
SUPPORT = ROOT / "tests/packedscan_simt"


@pytest.fixture(scope="module")
def literal_pair_vote(tmp_path_factory):
    build = tmp_path_factory.mktemp("pair-vote-literal")
    compiler = shutil.which("clang++")
    if not compiler:
        pytest.fail("literal decoder proof requires a C++20 compiler; install clang++ before running this suite")
    source = (NATIVE / "bi_gemm.cuh").read_text()
    helpers = source[source.index("__device__ __forceinline__ uint32_t t2_hscan16("):
                     source.index("template <int KQT>\nstruct Ld<F_TBE2, KQT>")]
    block = source[source.index("constexpr int T21_TAB_BYTES"):
                   source.index("template <int KQT>\nstruct Drv<Ld<F_TBE21")]
    (build / "decode.inc").write_text(helpers + block)
    harness = (SUPPORT / "harness.cpp").read_text()
    old = "L::dec_pair(d,raw,lane,j,paired,table,got);"
    assert harness.count(old) == 1
    harness = harness.replace(old, "L::dec_pair_ballot(d,raw,lane,j,paired,table,got);")
    (build / "harness.cpp").write_text(harness)
    exe = build / "harness"
    command = [compiler, "-std=c++20", "-O2", "-pthread",
               "-DBI_T21_PAIR_VOTE=1", "-DBI_T21_PACKSCAN2=1", "-DBI_T21_TILE128=1",
               "-DBI_T21_TABREG=1", "-DBI_T21_PACKSCAN4=0", "-DBI_T21_SCAN=0",
               "-DBI_T21_SCAN4=0", "-DBI_T21_ASYNC_X=0", "-DBI_T21_ROW_GEOMETRY=0",
               "-DBI_T21_WIDE128=0", "-DBI_T21_PIPE128=0",
               "-I" + str(SUPPORT), "-I" + str(build), "-I" + str(NATIVE),
               str(build / "harness.cpp"), "-o", str(exe)]
    compiled = subprocess.run(command, capture_output=True, text=True, timeout=90,
                              close_fds=False)
    (build / "compile.log").write_text(compiled.stdout + compiled.stderr)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    # Odd multiplication permutes every BF16 payload and mixes exponent digits.
    words = (np.arange(65 * 1024, dtype=np.uint32) * 40503 + 17).astype("<u2").reshape(65, 1024)
    assert np.unique(words).size == 65536
    encoded = codec_v21.encode_tensor("synthetic-all-bits-tail", words, dtype="BF16")
    return build, exe, words, encoded


def write_case(fixture, name, split):
    build, exe, words, encoded = fixture
    case = build / name
    case.mkdir()
    stream = bigemm_tbe21_ref.streams_for([encoded], split)
    for field, value in (("l1", stream.l1), ("ovf", stream.ovf), ("rowoff", stream.rowoff),
                         ("ckpt", stream.ckpt), ("rowp", stream.rowp.astype(np.int32)),
                         ("smb", stream.smb), ("cb", stream.cbtab), ("expected", words)):
        value.tofile(case / (field + ".bin"))
    if split > 1:
        assert np.count_nonzero(stream.ckpt & 31) > 0
    return case, [str(exe), str(case), str(stream.N), str(stream.K), str(split)]


@pytest.mark.parametrize("split", [1, 3, 8])
def test_literal_vote_matches_scalar_source_and_cursors(literal_pair_vote, split):
    case, command = write_case(literal_pair_vote, "split-" + str(split), split)
    result = subprocess.run(command, capture_output=True, text=True, timeout=240,
                            close_fds=False)
    (case / "run.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS words=66560 scalar=paired=source, cursors exact" in result.stdout


def test_corrupt_codebook_fails_original_word_oracle(literal_pair_vote):
    case, command = write_case(literal_pair_vote, "corrupt-codebook", 3)
    path = case / "cb.bin"
    original = path.read_bytes()
    path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    result = subprocess.run(command, capture_output=True, text=True, timeout=60,
                            close_fds=False)
    assert result.returncode == 3
    assert "mismatch" in result.stderr or "differs" in result.stderr
