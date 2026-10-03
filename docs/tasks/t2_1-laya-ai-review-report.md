# T2.1 — Independent Laya Label Review

**Status:** AI review completed; human review still required.  
**Input:** 100 Laya sentence/clause units from 79 pre-T0 Video Games reviews.  
**Reviewer:** one independent subagent pass using the original worksheet context and snapshot item titles as product-scope context. Prior Laya labels were treated as proposals, not truth.

## Results

- `8` agree with the existing Laya aspect/scope decision.
- `77` corrections (including newly proposed labels, removed labels, or scope corrections).
- `15` uncertain cases requiring adjudication.
- `59` units are product/accessory, packaging, shipping, or otherwise outside the game-aspect ontology.
- `32` units have proposed aspect labels; `32` are marked `proposed_needs_adjudication`.
- Aspect proposals: gameplay `17`, performance `7`, content/replay `5`, value `4`, story `2`, graphics `1`, multiplayer `1`.

## Systematic failures found

1. Laya over-predicted unrelated aspects for generic game criticism: “not remotely as fun” was labeled controls/multiplayer, and “error prone poorly programmed” was labeled controls/multiplayer. These belong to gameplay/performance.
2. A broken headset was labeled multiplayer; hardware failures are outside the game ontology.
3. A Bluetooth adapter sentence was assigned six game aspects despite describing an accessory and chat hardware.
4. Vague praise (“awesome”, “loved the game”, “good game”) was usually left unlabeled; this is preferable to inventing an aspect, but these cases remain adjudication candidates.
5. Evidence in the original Laya output covered entire clauses. Proposed evidence is narrowed to the supporting phrase where possible and uses source-text offsets.

## Recommendations

- Review all `uncertain` rows first, then the 77 corrections.
- Keep accessory, shipping, packaging, and hardware rows out of the game-aspect gold set.
- Do not treat this pass as gold labels or calibrated confidence. Lock labels only after human adjudication and the two-annotator agreement protocol.
