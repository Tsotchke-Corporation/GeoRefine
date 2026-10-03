#!/usr/bin/env python3
"""Build the serving-benchmark prompt pool (JSONL) from fixed chat/code questions and real
source files (read only).  Deterministic: same --src tree + seed -> same file.

classes
  chat    short chat / coding requests (no context), max_tokens cycles 256/512/1024
  ctx1k   ~1k-token code context + a coding question
  ctx8k   ~8k-token code context + a coding question
"""
import argparse
import json
import random
from pathlib import Path

CHAT = [
    "Write a Python function that parses an ISO-8601 duration string like 'P3DT4H12M' into seconds. Include unit tests with pytest and explain edge cases.",
    "Explain the difference between a mutex, a semaphore and a condition variable, with a short C++ example of each.",
    "Implement an LRU cache in TypeScript with O(1) get and put. Explain the design and give usage examples.",
    "I have a Flask app that becomes slow under load. Walk me through how to profile it and list the ten most likely causes with fixes.",
    "Write a bash script that finds the 20 largest files under a directory, skipping .git folders, and prints human-readable sizes.",
    "Explain how a B-tree index works in PostgreSQL and when a BRIN or GIN index would be better. Give concrete SQL examples.",
    "Write a Rust function that merges k sorted vectors into one sorted vector using a binary heap. Include tests and complexity analysis.",
    "What are the main causes of memory leaks in long-running Node.js services, and how do I find them in production?",
    "Implement Dijkstra's algorithm in Go for a graph given as an adjacency list, and return the shortest path, not just the distance.",
    "Write a detailed code review of this Python snippet and rewrite it idiomatically: `def f(l):\n  r=[]\n  for i in range(len(l)):\n    if l[i]%2==0: r.append(l[i]*l[i])\n  return r`",
    "Explain the CAP theorem with practical examples from Cassandra, MongoDB and etcd, and what trade-offs each makes.",
    "Write a SQL query that returns, for each customer, their three most recent orders and the running total of spend, then explain the window functions used.",
    "Design a rate limiter for a public REST API. Compare token bucket, leaky bucket and sliding window, and write a Redis-based implementation in Python.",
    "Write a React component for a searchable, paginated table that fetches data from an API, with loading and error states.",
    "Explain how gradient descent with momentum and Adam differ, with the update equations and a small NumPy implementation of each.",
    "Write a C function that reverses a singly linked list in place, then a recursive version, and discuss the stack usage of each.",
    "How do I set up a GitHub Actions workflow that runs tests on Python 3.10-3.13, caches pip, and publishes to PyPI on tags? Give the YAML.",
    "Explain what a race condition is using a bank-account example, then show how to fix it in Java with synchronized, locks and atomics.",
    "Write a Python asyncio program that downloads 100 URLs with at most 10 concurrent requests, retries with exponential backoff and reports failures.",
    "Summarise the key ideas of the transformer architecture for a software engineer: attention, positional encoding, layer norm and residuals.",
    "Write a Dockerfile for a production Django app with a multi-stage build, a non-root user, and gunicorn. Explain every line.",
    "Implement a trie in Python supporting insert, search, prefix search and delete, with tests.",
    "Explain the differences between processes and threads on Linux, including memory, scheduling and how fork and clone work.",
    "Write a Kotlin data class and repository layer for a todo app using Room, including DAO, entity and a migration example.",
    "My Kubernetes pod keeps getting OOMKilled. Explain how requests and limits work and give a step-by-step debugging plan.",
    "Write a function in JavaScript that deep-clones an object, handling Dates, Maps, Sets and circular references. Explain the approach.",
    "Explain how TLS 1.3 handshakes work step by step and what changed from TLS 1.2.",
    "Implement quicksort and mergesort in C++ with templates, compare their performance characteristics, and show a benchmark harness.",
    "Write a Python script that reads a large CSV (10 GB) and computes per-group statistics without loading it all into memory.",
    "Explain eventual consistency and how CRDTs achieve convergence. Implement a G-Counter and a PN-Counter in Python.",
]
QUESTIONS = [
    "Explain what this code does, section by section, and point out any bugs or risky patterns.",
    "Write unit tests covering the main functions in this code and explain what each test checks.",
    "Refactor the most complex function in this code for readability, and explain your changes.",
    "Summarise the design of this code for a new team member and list five improvements you would make.",
    "Identify performance bottlenecks in this code and propose concrete optimisations with code.",
]
MAXT = [256, 512, 1024]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source tree to draw code context from (read only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--n-ctx", type=int, default=40)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    files = sorted(p for p in Path(a.src).rglob("*") if p.suffix in (".cpp", ".c", ".h", ".py", ".cu")
                   and p.is_file() and 20000 < p.stat().st_size < 400000)
    out = []
    for i, q in enumerate(CHAT):
        out.append({"class": "chat", "max_tokens": MAXT[i % 3], "messages": [{"role": "user", "content": q}]})
    for cls, nchar in (("ctx1k", 3400), ("ctx8k", 27000)):
        for i in range(a.n_ctx):
            f = files[rng.randrange(len(files))]
            txt = f.read_text(errors="replace")
            s = rng.randrange(0, max(1, len(txt) - nchar))
            ctx = txt[s:s + nchar]
            q = QUESTIONS[i % len(QUESTIONS)]
            out.append({"class": cls, "max_tokens": MAXT[i % 3], "source": f"{f.relative_to(a.src)}@{s}",
                        "messages": [{"role": "user", "content": f"File `{f.name}`:\n```\n{ctx}\n```\n\n{q}"}]})
    Path(a.out).write_text("".join(json.dumps(x) + "\n" for x in out))
    print(len(out), "prompts ->", a.out)


if __name__ == "__main__":
    main()
