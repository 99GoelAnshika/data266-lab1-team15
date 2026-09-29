# Task 2 - Failure and Error Analysis

**Focal model:** BiGRU-Attention
**Evaluation population:** frozen official Yelp Polarity test split
**Decision threshold:** 0.50
**Review-set size:** 20 predetermined errors

This document summarizes the committed post-hoc qualitative error analysis. The 20 row identifiers were selected from the frozen prediction artifact before raw review text was accessed.

The review set contains exactly 5 confident false positives, 5 confident false negatives, 5 near-threshold errors, and 5 slice-specific errors.

**Scientific boundary:** these observations are descriptive only. They were not used to change model weights, architecture, hyperparameters, preprocessing, the decision threshold, robustness slice definitions, or checkpoint selection.

## 1. Frozen 20-example evidence table

| # | Category | Source row | True | Predicted | P(positive) | Effective length | OOV rate | Actual review-text snippet |
|---:|---|---:|---|---|---:|---:|---:|---|
| 1 | confident_false_positive | 29330 | negative | positive | 1.000000 | 21 | 0.000000 | Wow love the place and everything is very clean and new! Great place to come and relax worth a try! Cheers, Eric Van Nguyen Visited April 2012 |
| 2 | confident_false_positive | 14825 | negative | positive | 0.998535 | 153 | 0.013072 | Do you believe in Yin and Yang? The ancient Asian philosophy suggesting polar opposites are interrelated? If you don't, then you best start believing in it if you're headed to Flo's. The manifestation of Yin and Yang is clearly present...like 'in your fac... |
| 3 | confident_false_positive | 21331 | negative | positive | 0.998047 | 26 | 0.000000 | This place is only awesome if you are one of three things: 1. an Ohio State fan, 2. a metrosexual, or 3. someone trying to hook up with an Ohio State fan or metrosexual...Everyone else, do yourself a favor and go to any one of the many hundreds of awesome bars... |
| 4 | confident_false_positive | 5752 | negative | positive | 0.997559 | 221 | 0.004525 | NOTE: This was a 4-star review, but the food quality and ESPECIALLY customer service have gone down the tubes. See update below. Complaining I can't find a good meatball sub in Phoenix, I was referred to Santisi Brothers. I was told that the meatballs are sti... |
| 5 | confident_false_positive | 8426 | negative | positive | 0.997559 | 232 | 0.004310 | Saturday / Sunday AYCE brunch In true las vegas fashion, you get a flat rate to eat your heart out. For Strip food, main menu items seem reasonably priced and the brunch is cheap at $29.99. There are a few different flavors to choose from for the All-Yo... |
| 6 | confident_false_negative | 22807 | positive | negative | 0.000229 | 50 | 0.000000 | EDIT: They really did change the service up since I last posted this. Horrible service. Used to be my favorite pizza in the city (at a reasonable price), but I'm rethinking that. We just had an altercation with a server who refused to split a check when... |
| 7 | confident_false_negative | 30793 | positive | negative | 0.000782 | 43 | 0.000000 | This place is so much better since they changed owners. My wife and I went when it was the old owners, it was terrible. We waited forever and the food never came before we walked out. People were served before us that walked in after and my wife actually go... |
| 8 | confident_false_negative | 30958 | positive | negative | 0.001561 | 193 | 0.000000 | Look, we all know Cox sucks. In fact they are a terrible business and their practices are laughable. However they are a necessary evil if you want Internet that isn't garbage that can handle streaming/gaming. The way they handle issues over the phone is worse... |
| 9 | confident_false_negative | 22451 | positive | negative | 0.001782 | 116 | 0.000000 | The food is crap. I'm not trying to be mean, but it really is horrible. I'd rather eat one of those Tornado things from Circle K for dinner. Also, I don't appreciate the waitress telling me everything is great when everything is absolutely not great. What kind... |
| 10 | confident_false_negative | 25329 | positive | negative | 0.002132 | 92 | 0.010870 | Im with Kimberly B...this is a card for people rebuilding credit. Mine got tanked due to a divorce. I have been rebuilding, and started with a $300 limit, and it is now at $1250. I do not carry a balance, ever. If you need to carry a balance, then you haven't... |
| 11 | near_threshold_error | 12415 | negative | positive | 0.500000 | 59 | 0.000000 | My friends and I were in the mood for some cheap sushi..and since I was with 5 guy friends, I was not surprised they chose an all-you-can-eat place. As per my profile, I LOVE AYCE sushi. However, Makino doesn't quite satisfy this AYCE sushi lover. The sushi ro... |
| 12 | near_threshold_error | 16620 | negative | positive | 0.500000 | 37 | 0.000000 | blush is a cool place to hang out.. idk if it's a lounge. felt like it.. but with a club crowd? there's not enough places to chill and a million ppl walking around and around each other.. music was very typical vegas which isn't a favorite of mine but i deal..... |
| 13 | near_threshold_error | 31278 | negative | positive | 0.500000 | 17 | 0.000000 | How hard is it to get a 2 donut order right?? One chocolate glazed and one strawberry glazed donut. That's IT. Apparently pretty hard. |
| 14 | near_threshold_error | 37323 | negative | positive | 0.500000 | 26 | 0.000000 | They have a good space but I don't think their using it wisely... The food is okay for a bar but my martini was very good. Adding a bit more seating would be a good idea because when I went there was not enough seating for watching a band. May go back for an e... |
| 15 | near_threshold_error | 18639 | positive | negative | 0.499756 | 7 | 0.000000 | You really need a whole day to see this centre. The gardens are spectacular, as are the views. |
| 16 | slice_specific_error | 12480 | negative | positive | 0.997070 | 11 | 0.000000 | my husband had an omelette that was good. i had a blt, a little on the small side for $10, but bacon was great. Our server was awesome! |
| 17 | slice_specific_error | 20203 | negative | positive | 0.997559 | 50 | 0.000000 | Not all that. This joint has some of the best reviews short of Joel Robuchon but the fact is it's not much more than a glorified protein-buffet. The quality of meat is far from Prime; the salad bar is not bad; the wine list is fair; the service is acceptabl... |
| 18 | slice_specific_error | 6611 | positive | negative | 0.003111 | 121 | 0.008264 | Harlow's gives me diarrhea. Every time. Every single time. You know what, though? I don't care because Harlow's cuts my hangover directly in half and I feel twice as good when I leave. If I get me some diarrhea, that's okay. It's a price I'm willing to pay. \... |
| 19 | slice_specific_error | 37771 | negative | positive | 0.997559 | 31 | 0.000000 | I have to say I love cafe rio! Their food is simple yet delicious, the service is great as well as consistent, and my appetite and tummy are always satisfied. However I've been to this location 3 different times and have not experienced any of these.. Even tho... |
| 20 | slice_specific_error | 26587 | negative | positive | 0.990723 | 256 | 0.050781 | On the whole, this restaurant does give a new patron a great slice of European cuisine and it is worth noting that if you've never been here before that you should try it out because most of the food as it is now is delicious, but having been here in the past... |

The table contains actual text excerpts from the frozen snapshot; the full raw review text and its SHA256 are preserved in `outputs/manual_error_analysis/manual_review_text_snapshot.json`.

## 2. Existing Notebook 06 qualitative interpretation

The following narrative is carried forward from the already committed Notebook 06 analysis. It is not a new test-set analysis.

This notebook performs the required qualitative review of **20 frozen official-test examples** for the BiGRU-Attention sentiment classifier.

The review set contains exactly:

- 5 confident false positives
- 5 confident false negatives
- 5 near-threshold errors
- 5 slice-specific errors

The row selection was frozen and pushed to Git **before review text was accessed**. The raw review-text evidence was subsequently frozen and pushed in a separate commit before qualitative interpretation was authored.

This notebook is strictly post-hoc and descriptive. No observation in this notebook is used to modify model weights, architecture, hyperparameters, decision threshold, preprocessing, robustness-slice definitions, or checkpoint selection.
## Interpretation boundary

The purpose of this notebook is to understand **why representative errors may have occurred**, not to claim causal attribution from model internals.

Therefore:

- phrases such as *consistent with*, *plausible failure mode*, and *may reflect* are used deliberately;
- no attention weights, saliency maps, or token-level causal explanations are available here;
- apparent label-text disagreements are documented as **possible annotation or source-context issues**, not automatically declared mislabeled;
- all conclusions are based on the already-frozen test predictions and already-frozen raw review snapshot;
- these observations must not feed back into model, threshold, hyperparameter, or slice tuning.
## Review fields

For every selected example, the notebook retains:

- frozen test/source row identifier;
- true dataset label;
- BiGRU-Attention prediction;
- positive-class probability;
- wrong-class confidence;
- effective model-visible length;
- OOV count and OOV rate;
- frozen robustness-slice memberships;
- exact raw review text;
- a manually authored primary failure-mode hypothesis;
- a qualitative interpretation;
- a note about apparent agreement between the visible text and dataset label.

The qualitative fields are interpretations of the evidence, not additional model outputs.
# 1. Confident False Positives

These are negative-labeled examples for which BiGRU-Attention predicted the positive class with very high confidence.

The five cases reveal several distinct phenomena rather than one single failure pattern:

1. **Possible label-text disagreement** - one review is visibly positive despite the negative dataset label.
2. **Sarcasm** - negative service criticism is expressed through superficially positive language.
3. **Scope/entity confusion** - positive terms describe conditions or alternative venues rather than the reviewed venue.
4. **Review-history conflict** - a short negative update is combined with a much longer older positive review.
5. **Mixed aspect sentiment** - negative food judgments coexist with substantial praise for service, drinks, and atmosphere.
# 2. Confident False Negatives

These are positive-labeled examples for which BiGRU-Attention predicted the negative class with very high confidence.

The cases are dominated by **document-level contrast**:

- a possible update/rating mismatch;
- a formerly bad experience followed by a positive current judgment;
- a long complaint narrative followed by a decisive positive resolution;
- severe criticism of one aspect despite an overall positive rating;
- positive endorsement expressed through language about financial problems, fees, blame, and mistakes.

These examples show why document-level sentiment cannot always be inferred from the local density of positive and negative words.
# 3. Near-Threshold Errors

The five examples nearest the frozen 0.50 decision threshold are qualitatively different from the confident errors.

They contain:

- genuine mixed sentiment;
- informal and ambivalent language;
- sarcasm expressed with very few explicit polarity words;
- mild criticism balanced by some praise;
- an extremely short positive review with very little evidence.

The near-threshold behavior is therefore consistent with examples where the available model-visible evidence is weak, balanced, or pragmatically indirect.
# 4. Slice-Specific Errors

Exactly one disjoint error was frozen for each robustness slice:

1. short review;
2. medium review;
3. long review;
4. contains-negation;
5. high-OOV-rate.

The high-OOV example is especially informative because its effective model-visible length is exactly **256 tokens**, the configured maximum sequence length. Its review begins with extensive positive historical description, while the explanation for the low rating appears much later. This makes **sequence truncation a concrete, evidence-supported limitation of the input representation**: later text cannot influence the classifier once it lies beyond the fixed model-visible sequence.

The other slice examples reinforce the roles of short-context ambiguity, contrast, mixed sentiment, and negation.
# Cross-Case Findings

Across the 20 predetermined examples, the most important qualitative patterns are:

### 1. Mixed sentiment and aspect conflict
Several errors combine strong positive and negative statements in the same review. A single document-level label can depend on which aspect the reviewer ultimately weights most heavily.

### 2. Temporal and discourse reversal
Multiple reviews describe a bad past experience before a positive current conclusion, or begin positively before reversing later. Correct classification requires understanding discourse structure rather than treating all sentiment-bearing phrases equally.

### 3. Sarcasm, rhetorical language, and sentiment scope
Some false positives contain positive words whose pragmatic meaning is negative, conditional, sarcastic, or directed toward another entity.

### 4. Rating/text or update ambiguity
A small number of reviews appear difficult to reconcile with their frozen label using the visible text alone. These are documented as possible label-text or review-history mismatches rather than automatically treated as model mistakes.

### 5. Short-input uncertainty
Several near-threshold or slice-specific reviews contain very little model-visible evidence. Short inputs can make the posterior probability unstable around the fixed threshold.

### 6. Fixed-length truncation
The high-OOV slice example has an effective length of exactly 256 tokens while the original review is much longer. The decisive negative discussion occurs late in the document. Since the model cannot consume tokens beyond the configured maximum length, this is a concrete representation limitation rather than merely an abstract hypothesis.

### 7. Negative lexical density can obscure a positive final stance
Several false negatives contain many strongly negative phrases even though the reviewer ultimately recommends the business, preserves a positive rating, or explicitly states a positive final opinion.
# Scientific Boundary

This manual review is **retrospective error characterization**.

It does **not** establish causal explanations of the BiGRU-Attention model's internal reasoning, and it does not constitute a new model-selection experiment.

Most importantly, because these examples come from the already-consumed official test set:

- no model may be retrained because of these observations;
- no hyperparameter may be changed because of these observations;
- the 0.50 threshold must not be changed because of these observations;
- robustness-slice definitions must not be changed because of these observations;
- no new checkpoint may be selected using these observations.

The appropriate use of this analysis is to document limitations, describe representative failure modes, and motivate possible future work on a separate future dataset or validation protocol.

## 3. Evidence provenance

- Frozen prediction source SHA256: `9D912CFE0E743D1468D3B41BA64ED93E9F54B5E556021A1C0D87D05D11072BC4`
- Selection frozen before raw-text access: `True`
- Snapshot selection frozen before text access: `True`
- Selection-manifest commit: `548a175e27326046dee0fd1203f0341147dd4cee`
- Dataset revision: `bbf1c97a1f0cf005e5aded43839fd814654a1557`
- Full analysis notebook: `task2_sentiment/Anshika_Goel/notebooks/06_manual_error_analysis.ipynb`

Future improvements suggested by these errors must be evaluated using a separate future validation/test protocol rather than by retuning against the already-consumed official test set.
