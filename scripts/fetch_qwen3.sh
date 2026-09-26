#!/bin/bash
# Download Qwen3-30B-A3B (61GB) from ModelScope.
#
# Why ModelScope: the HF CDN is flaky from this network — transfers stall
# silently and the `hf` CLI's 10s read timeout aborts + restarts from
# scratch, so it never finishes a 4GB shard. Measured sustained rates:
#   HF CDN        2.85 MB/s   (with frequent stalls)
#   hf-mirror     3.29 MB/s
#   ModelScope    5.23 MB/s   <- used here
#
# ModelScope's copies are byte-identical to HF's (sha256 prefixes match),
# so the partial shards already fetched from HF are resumed, not discarded.
#
# Properties: 6 parallel shards, resumable (-C -), stall-detecting
# (abort under 100KB/s for 30s and retry), sha256-verified, and detached
# (setsid) so it outlives the driving session.
set -u

REPO="Qwen/Qwen3-30B-A3B"
OUT="/home/kitty/plastic-infer/models/qwen3-30b-a3b-hf"
BASE="https://modelscope.cn/models/$REPO/resolve/master"
META="/home/kitty/plastic-infer/scripts/qwen3_shards.txt"
PAR=6

mkdir -p "$OUT"

log() { echo "[$(date +%H:%M:%S)] $*"; }

fetch_shard() {
  local f="$1" size="$2" sha="$3"
  local url="$BASE/$f" dest="$OUT/$f"

  if [ -f "$dest" ] && [ "$(stat -c%s "$dest")" = "$size" ] \
     && [ "$(sha256sum "$dest" | cut -d' ' -f1)" = "$sha" ]; then
    log "SKIP $f (verified)"; return 0
  fi

  local attempt stall=0 last=-1 now
  for attempt in $(seq 1 300); do
    now=$(stat -c%s "$dest" 2>/dev/null || echo 0)
    [ "$now" -gt "$size" ] && { rm -f "$dest"; now=0; }
    curl -L -C - -s -o "$dest" \
         --connect-timeout 30 --max-time 3600 \
         --speed-limit 100000 --speed-time 30 \
         --retry 5 --retry-delay 10 --retry-all-errors \
         "$url"
    now=$(stat -c%s "$dest" 2>/dev/null || echo 0)
    if [ "$now" = "$size" ] \
       && [ "$(sha256sum "$dest" | cut -d' ' -f1)" = "$sha" ]; then
      log "OK $f ($((size/1048576))MB, $attempt attempts)"; return 0
    fi
    if [ "$now" = "$last" ]; then
      stall=$((stall + 1))
      log "stall $stall on $f at $((now/1048576))/$((size/1048576))MB"
      [ "$stall" -ge 3 ] && { log "restart $f from 0"; rm -f "$dest"; stall=0; }
    else
      stall=0
    fi
    last="$now"
    [ "$now" = "$size" ] && { rm -f "$dest"; last=-1; }   # size ok, hash bad
  done
  log "FAIL $f"; return 1
}

export -f fetch_shard
export -f log
export OUT BASE

log "=== 16 shards, $PAR parallel, from ModelScope ==="
# shellcheck disable=SC2016
awk '{print $1, $2, $3}' "$META" \
  | xargs -P "$PAR" -n 3 bash -c 'fetch_shard "$@"' _
log "=== shards done ==="

# Small files (config / tokenizer / index) — tokenizer.json is required by
# the CLI's --prompt path.
for fn in config.json generation_config.json tokenizer.json \
          tokenizer_config.json vocab.json merges.txt \
          model.safetensors.index.json; do
  if [ ! -s "$OUT/$fn" ]; then
    curl -sL --connect-timeout 30 --max-time 600 --retry 5 --retry-delay 5 \
         --retry-all-errors -o "$OUT/$fn" "$BASE/$fn" \
      && log "fetched $fn"
  fi
done

log "=== ALL DONE: $(ls "$OUT"/*.safetensors 2>/dev/null | wc -l)/16 shards ==="
