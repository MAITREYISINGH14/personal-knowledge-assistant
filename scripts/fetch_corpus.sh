#!/usr/bin/env bash
# Downloads the public PDFs for the PKA test corpus into data/corpus/papers/.
# Run from the project root:   bash scripts/fetch_corpus.sh
# Safe to re-run: files that already downloaded correctly are skipped.

set -u
DEST="data/corpus/papers"
mkdir -p "$DEST"
UA="pka-corpus-fetch/1.0 (personal research project)"

# name | url | category
DOCS=(
  # --- Long (30+ pages) ---
  "nist_ai_risk_management_framework|https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf|long"
  "nist_generative_ai_profile|https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf|long"
  "survey_of_large_language_models|https://arxiv.org/pdf/2303.18223|long"
  # --- Medium (roughly 10-25 pages) ---
  "rag_survey_gao_2023|https://arxiv.org/pdf/2312.10997|medium"
  "rag_original_lewis_2020|https://arxiv.org/pdf/2005.11401|medium"
  "raptor_tree_retrieval_2024|https://arxiv.org/pdf/2401.18059|medium"
  "lost_in_the_middle_2023|https://arxiv.org/pdf/2307.03172|medium"
  "dense_passage_retrieval_2020|https://arxiv.org/pdf/2004.04906|medium"
  "sentence_bert_2019|https://arxiv.org/pdf/1908.10084|medium"
  "attention_is_all_you_need_2017|https://arxiv.org/pdf/1706.03762|medium"
)

ok=0; skipped=0; failed=()
for entry in "${DOCS[@]}"; do
  IFS="|" read -r name url cat <<< "$entry"
  out="$DEST/$name.pdf"
  if [[ -s "$out" ]] && head -c 4 "$out" | grep -q "%PDF"; then
    echo "  skip  $name (already downloaded)"; skipped=$((skipped+1)); continue
  fi
  echo "  get   $name  [$cat]"
  if curl -fsSL --retry 2 --max-time 120 -A "$UA" -o "$out.part" "$url" \
     && head -c 4 "$out.part" | grep -q "%PDF"; then
    mv "$out.part" "$out"; ok=$((ok+1))
  else
    rm -f "$out.part"; failed+=("$name  <-  $url")
  fi
  [[ "$url" == *arxiv.org* ]] && sleep 3   # be polite to arXiv
done

echo
echo "Downloaded: $ok   Already present: $skipped   Failed: ${#failed[@]}"
if (( ${#failed[@]} )); then
  echo "These failed (open the link in a browser, save the PDF into $DEST with the name shown):"
  printf '   %s\n' "${failed[@]}"
fi
echo
echo "Next: python -m scripts.inventory_corpus"
