# ADR-0021: Non-game items stay in ranking but get no game-aspect claims

Status: Accepted on 2026-11-16. The project owner asked Claude to make this decision. Refines [ADR-0006](0006-category-and-entity-scope.md).

## Context

ADR-0006 keeps game aspects for software game entities. It marks consoles,
controllers, accessories, and similar items as out of scope. The snapshot
never applied that rule: `entity_scope` is `software_game` for all 49,760
items. The Video Games catalog also holds store credit, which ADR-0006 does
not name: gift codes, currency and points cards, memberships, and
subscriptions.

B5 and H0 recommend some of these items often. Two membership and currency
claims had about 115,000 views between them. ADR-0020 verification rejected
their gameplay claims, because their reviews talk about code delivery and
redemption. Removing these items from the catalog would change the locked
temporal protocol and the comparability of every T6 accuracy result.

## Decision

1. **Ranking.** The ranking catalog and the evaluation protocol do not
   change. Non-game items stay rankable, because buying a membership or a
   points card is a real purchase in this category. Accuracy results stay as
   reported.
2. **Explanations.** Game-aspect claims apply to game items only, which
   enforces ADR-0006 at explanation time. A non-game recommendation is
   `unavailable` with refusal reason `non_game_item:<class>`, and shows no
   aspect claim.
3. **Classes.** An item is non-game when either rule matches. Store credit is
   checked first.
   - `store_credit`: the title matches a fixed list of terms for gift cards,
     gift codes, currency and points, prepaid and game-time cards,
     memberships, and subscriptions (`trustrec.data.item_scope`).
   - `hardware_accessory`: at least 5 of the item's pre-cutoff reviews have
     curated LLM labels, and at least half of those are `out_of_scope`. Under
     the labeler prompt, that means the reviews mostly discuss consoles,
     controllers, accessories, or cables.
4. **Record.** The classification is written once per snapshot and cutoff,
   with its rules, counts, and source hashes. A rule change needs a new
   decision.

5. **Reporting.** Accuracy reports also give a game-target slice: test
   targets on game items only, with unchanged candidates and rankings
   (`scripts/build_game_target_slice.py`).

## Consequences

- Explanation coverage now counts non-game recommendations as `unavailable`.
  Reports give their share separately.
- DLC, expansions, and season passes (for example "Booster Course Pass") stay
  game items, because game aspects apply to them.
- Store credit with a game name in the title (for example "Microsoft Points
  for Call of Duty") is still store credit.
- The first game-target slice reverses the accuracy ranking: B5 is worst and
  B3 PPR best on game targets (see T6.4). The ADR-0018 choice of the
  explained recommender assumed the all-target results, and the owner should
  revisit it.
