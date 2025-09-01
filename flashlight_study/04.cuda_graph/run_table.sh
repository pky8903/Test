#!/usr/bin/env bash
# run_bench.sh
# Run ./build/my_app over all parameter combinations and aggregate results into a CSV.

set -euo pipefail

BIN="./build/my_app"
ITER=20
OUTCSV="results.csv"

# Parameter ranges
Ns=(1 4)
Cs=(1 4)
Hs=(256 512 1024 2048 4096)

# Write CSV header
echo "N,C,H,W,iter,Eager time (ms),Eager time (ms/iter),Graph time (ms),Graph time (ms/iter),Speedup" > "$OUTCSV"

# Robust extractors (tolerate leading spaces and variable spacing)
extract_total() { # $1=line label (Eager|Graph), stdin=full output
  grep -m1 -E "[[:space:]]*$1[[:space:]]+time" \
  | sed -nE 's/.*[Tt]ime[[:space:]]*:[[:space:]]*([0-9.+-eE]+).*/\1/p'
}

extract_per_iter() { # $1=line label (Eager|Graph), stdin=full output
  grep -m1 -E "[[:space:]]*$1[[:space:]]+time" \
  | sed -nE 's/.*\(([[:space:]]*)([0-9.+-eE]+)[[:space:]]*ms\/iter\).*/\2/p'
}

for N in "${Ns[@]}"; do
  for C in "${Cs[@]}"; do
    for H in "${Hs[@]}"; do
      W="$H"
      echo "[INFO] Running: $BIN $N $C $H $W $ITER" 1>&2

      out="$("$BIN" "$N" "$C" "$H" "$W" "$ITER" 2>&1 || true)"

      # Parse totals and per-iter; if total missing, derive from per-iter * ITER
      eager_total=$(printf "%s" "$out" | extract_total "Eager" || true)
      eager_per_it=$(printf "%s" "$out" | extract_per_iter "Eager" || true)
      graph_total=$(printf "%s" "$out" | extract_total "Graph" || true)
      graph_per_it=$(printf "%s" "$out" | extract_per_iter "Graph" || true)

      if [[ -z "${eager_total:-}" && -n "${eager_per_it:-}" ]]; then
        eager_total=$(awk -v p="$eager_per_it" -v it="$ITER" 'BEGIN{printf("%.9g", p*it)}')
      fi
      if [[ -z "${graph_total:-}" && -n "${graph_per_it:-}" ]]; then
        graph_total=$(awk -v p="$graph_per_it" -v it="$ITER" 'BEGIN{printf("%.9g", p*it)}')
      fi

      # Validate parsed values
      if [[ -z "${eager_total:-}" || -z "${graph_total:-}" || -z "${eager_per_it:-}" || -z "${graph_per_it:-}" ]]; then
        echo "[WARN] Failed to parse times for N=$N C=$C H=$H W=$W. Raw output follows:" 1>&2
        echo "$out" 1>&2
        continue
      fi

      # Compute speedup = Eager total / Graph total
      speedup=$(awk -v e="$eager_total" -v g="$graph_total" 'BEGIN{
        if (g==0) {print "inf"; exit}
        printf("%.6f", e/g)
      }')

      # Append CSV row
      printf "%d,%d,%d,%d,%d,%s,%s,%s,%s,%s\n" \
        "$N" "$C" "$H" "$W" "$ITER" \
        "$eager_total" "$eager_per_it" "$graph_total" "$graph_per_it" "$speedup" \
        >> "$OUTCSV"
    done
  done
done

echo
echo "=== Summary Table (aligned) ==="
if command -v column >/dev/null 2>&1; then
  awk 'BEGIN{FS=","; OFS="\t"} {print $1,$2,$3,$4,$5,$6,$7,$8,$9,$10}' "$OUTCSV" | column -t -s $'\t'
else
  cat "$OUTCSV"
fi

echo
echo "[DONE] CSV saved to: $OUTCSV"

