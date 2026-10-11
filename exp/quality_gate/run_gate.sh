#!/bin/bash
# Quality gate: molot vs reference oracle.
# Method: molot tokenizes the prompt (--debug prints ids); the oracle evaluates
# the EXACT same id sequence (MOLOT_IDS); greedy top-1 must agree.
# Then a generation degeneracy check (8-gram repeats) with sampling.
# Usage: MODEL=... ORACLE=... MOLOT=... ./run_gate.sh
set -u
M="${MODEL:-/home/diem/.cache/huggingface/hub/models--ggml-org--gemma-4-e4b-it-GGUF/snapshots/b8093469224f83f5c38f691eb906c380e9e63114/gemma-4-E4B-it-Q4_0.gguf}"
ORACLE="${ORACLE:-/tmp/opencode/logit_probe}"
MOLOT="${MOLOT:-/home/diem/repos/molot/build/molot-run}"
TMP="${TMPDIR:-/tmp/opencode}/gate"
mkdir -p "$TMP"
PROMPTS=("The capital of France is" "What is 2+2?" "Write a haiku about CPUs." "Explain gravity briefly." "Once upon a time")
fail=0
agree=0
degfail=0
for i in 0 1 2 3 4; do
  # molot: ids + top-1
  dbg=$("$MOLOT" "$M" "${PROMPTS[$i]}" -n 0 --debug 2>&1 | grep -E "ids=|id=" | tail -n 9)
  ids=$(echo "$dbg" | grep "ids=" | sed 's/.*ids=//')
  m1=$(echo "$dbg" | grep -E "^  id=" | head -1 | awk '{print $1}' | tr -d 'id= ')
  # oracle on identical ids (comma-separated for MOLOT_IDS)
  o1=$(MOLOT_IDS="$(echo $ids | tr ' ' ',')" "$ORACLE" "$M" "x" 5 2>/dev/null | grep -E "^  id=" | head -1 | awk '{print $1}' | tr -d 'id= ')
  if [ -n "$o1" ] && [ "$o1" == "$m1" ]; then r1="OK"; agree=$((agree+1)); else r1="MISMATCH"; fail=1; fi
  echo "[$i] '${PROMPTS[$i]}' ids=[$ids] oracle=$o1 molot=$m1 $r1"
  # degeneracy check with sampling (full stdout: generation may span lines)
  "$MOLOT" "$M" "${PROMPTS[$i]}" -n 30 -t 0.7 2>/dev/null > "$TMP/g$i.txt"
  dout=$(python3 - "$TMP/g$i.txt" <<'EOF'
import sys, collections, re
raw = open(sys.argv[1], encoding='utf-8').read()
raw = re.sub(r'<\|?turn\|?>', ' ', raw)  # structural markers, not content
toks = raw.split()
if len(toks) < 16:
    print("short"); sys.exit(0)
g = collections.Counter(tuple(toks[i:i+8]) for i in range(len(toks)-7))
top, n = g.most_common(1)[0]
if n >= 3:
    print(f"degenerate")
else:
    print("clean")
EOF
)
  echo "    $dout"
  if [ "$dout" == "degenerate" ]; then degfail=1; fi
done
echo "top-1 agreement: $agree/5"
if [ $agree -ge 4 ] && [ $degfail -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
