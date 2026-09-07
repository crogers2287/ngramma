# Independent full-engine audit: eight of eight completed conditions

**Eight of eight conditions pass the independent raw-capture audit. No blocker found.** This includes unmodified, zero, and both signs of the three preselected magnitudes.

I read the saved raw captures and overlays only; no model forward, shard read, service change, or runtime-source modification was performed. The final `response-with-engine.json` SHA256 is `c9ceda821665811f3fbd58b0900a90716a36d3a598f01d53ee5ca9fbcd46ba1e`; the final progress snapshot SHA256 is `22e3650f4f7a7531c1a5349abf8650e684cffa566ecfc5471cc2a30c8e0165f1`; the local response SHA256 is `cae579594b721fde25af2c49bbe351cf4774d17c4d84de6b4c4b503b429abe04`.

## Independently checked controls

For unmodified, zero, `minus-12`, `plus-12`, `minus-10`, `plus-10`, `minus-8`, and `plus-8`, I independently reconstructed logical tensors from capture metadata and checked:

- Actual logits and tensor-metadata file hashes match each recorded capture summary, which matches the progress record. All eight use the same lens hash, capture-script hash, model identity, tokens, and CPU configuration. The lens hash and unmodified logits hash also match the pinned experiment-003 baseline recorded in the local response.
- Each overlay file hash matches its local response point. Every gathered activation byte equals the unmodified memory with exactly the selected row replacement at `[6,8]`; the layer-0 hidden tensor remains byte-identical.
- The complete reconstructed PLE output hash matches both the progress record and the corresponding local native response stage. This is full-tensor equality, not a single scalar comparison.
- The first six token-position logits remain byte-identical to baseline in all conditions. Each reported changed-value count and fixed-margin delta was recomputed from raw logits.

The unmodified, zero-overlay, and both `2^-12` conditions have literally identical logits bytes and SHA256, resolving the earlier concern that NumPy numeric equality alone cannot establish byte equality. Their PLE output hashes also equal baseline. The nonzero `2^-12` overlays change all 160 row coordinates, as established by the local audit, so these are actual row edits with unchanged observed encoded/projection/model outputs at those sampled steps.

## Completed finite responses

| Condition | Changed argmax positions (zero-based) | Changed logit positions | Fixed final margin | Margin change |
|---|---|---|---:|---:|
| Unmodified / zero / ±2^-12 | None | None | 8.437328338623047 | 0 |
| −2^-10 | 8, 9, 10 | 6 through 16 | 5.360069274902344 | −3.077259063720703 |
| +2^-10 | 10 | 6 through 16 | 4.943431854248047 | −3.493896484375 |
| −2^-8 | 10 | 6 through 16 | 7.433195114135742 | −1.0041332244873047 |
| +2^-8 | 10 | 6 through 16 | 12.739151000976562 | +4.301822662353516 |

The negative `2^-10` edit changes 2,731,520 logit values; the positive edit changes 2,731,519. Both shrink the margin between the two token IDs fixed from the unmodified final-position top two. The margins remain positive. All position comparisons condition on the same provided 17-token sequence; they are prefill next-token argmax comparisons, not independently generated continuations. No sampled autoregressive generation or answer-quality assessment was performed.

The local replay has already established that `±2^-10` changes only two FP16 activation scales, with integer codes unchanged. In one affected block the two directions produce the same scale increase because of tied maxima; in the other they produce opposite changes. That explains why the encoded perturbations are not opposite vectors. The full-engine measurements establish downstream sensitivity to these finite edits, with live routing and attention selection. They do not identify which individual route or subsequent numerical threshold contributes most to the large final response, because route selections and intermediate downstream layers were not captured for causal decomposition.

The smooth derivative belongs to a different FP64 local scalar, not this full-engine margin. A negative margin response in both directions neither disproves that smooth derivative's correctness for its own function nor establishes a deployed derivative. Shrinking a chosen margin is not by itself harmful or beneficial task behavior. Likewise, changed prefill argmax positions do not establish improved recall, useful learning, robustness, or usable training directions.

## Completion and scope

Both `2^-8` captures pass the same gathered replacement, unchanged earlier hidden/prefix logits, full PLE hash, logits count/margin, and shared lens/configuration checks. Each changes 2,731,519 logit values and one prefill argmax position. The positive edit increases the fixed final margin, while the negative edit decreases it. This is a finite response at the next selected magnitude, not evidence that the positive direction improves answers.

The final artifact binds the original local response hash. Its driver/capture-source and runtime-library hashes were independently compared with the current files and match. All eight capture summaries were rechecked against their actual logits and metadata hashes at completion. All selected conditions are retained; none is pending.

Measured scope remains one existing trigram row, one fixed direction, one 17-token fresh prefill, one CPU configuration, and three selected nonzero magnitudes plus zero/unmodified controls. The experiment supports exact local-to-engine PLE correspondence and finite downstream sensitivity. It does not qualify training, a serving derivative, autoregressive generation behavior, answer quality, or generalization to other rows or directions.
