# Task 1 Sequence Model Failure Analysis - Anshika Goel

## Method

The three cases below were selected from the model's actual generated outputs in `outputs/samples/generated_samples.json`. Each quoted passage is a verbatim excerpt from the corresponding continuation. No failure example was invented or manually corrected.

## Failure Case 1: Sentence-Level Repetition Loop

- Sample ID: `sample_001`
- Prompt: `Once upon a time`
- Decoding method: Greedy
- Temperature: Not applicable
- Repeated 4-gram rate: `0.543260`
- Source: `outputs/samples/generated_samples.json`

Verbatim excerpt:

> She said they were going to the park to play with it. She said they were going to the park to play with it. She said they were going to the park to play with it. She said they were going to the park to play with it.

### Observation

The model initially produces a plausible story about Lily and her parents but becomes trapped in an exact sentence-level loop. The repeated 4-gram rate of approximately 54.33% confirms that more than half of the continuation's 4-gram occurrences duplicate an earlier 4-gram.

### Likely Cause

Greedy decoding always selects the highest-probability next character. Once the model enters a locally probable phrase, it has no randomness or repetition control to help it leave that sequence. The 256-character context window may also reinforce the repeated sentence as it increasingly dominates the recent context.

### Testable Improvement

Add a decoding-time repetition penalty or block previously generated 4-grams. Generate the same prompts with and without this constraint, then compare repeated 4-gram rate and human-rated coherence. The improvement would be supported if repetition decreases without introducing major grammatical degradation.

## Failure Case 2: Pathological Word Repetition

- Sample ID: `sample_005`
- Prompt: `Once upon a time`
- Decoding method: Temperature sampling
- Temperature: `1.0`
- Repeated 4-gram rate: `0.396378`
- Source: `outputs/samples/generated_samples.json`

Verbatim excerpt:

> she found a box of blue blue blue blue blue balls.
>
> She brought it to the blue blue blue blue blue blue blanket! The blue blue blue blue blue blue, near a tree and carrots.

### Observation

The continuation repeatedly emits the word `blue`, producing locally grammatical fragments but no meaningful narrative progression. Its repeated 4-gram rate is approximately 39.64%, the highest among the temperature-1.0 sampled outputs.

### Likely Cause

Because the model predicts one character at a time, a short and common word can become a highly probable self-reinforcing pattern. Standard temperature sampling changes the probability distribution but does not explicitly discourage reuse of recently generated character sequences.

### Testable Improvement

Use unlikelihood training or a decoding penalty targeted at recently repeated character n-grams. Re-run generation at temperature 1.0 with the same seed and compare the frequency of repeated words, repeated 4-gram rate, and story completeness.

## Failure Case 3: Grammar and Semantic Coherence Breakdown

- Sample ID: `sample_008`
- Prompt: `Once upon a time`
- Decoding method: Temperature sampling
- Temperature: `1.3`
- Repeated 4-gram rate: `0.171026`
- Source: `outputs/samples/generated_samples.json`

Verbatim excerpt:

> The nlight was nothing happened, hugged and ran off for bed.
>
> Moral.
>
> Joey was so happy to have opened a bowl very year of.
> Before exhauster his 3 years old and he finished, knowing she was excited.

### Observation

This sample has less mechanical repetition than the first two cases, but it contains malformed words, broken syntax, abrupt transitions, inconsistent pronouns, and unrelated events. Phrases such as `The nlight was nothing happened` and `Before exhauster his 3 years old` are not grammatically or semantically coherent.

### Likely Cause

A temperature of 1.3 flattens the next-character probability distribution, increasing the chance of selecting unlikely characters. Character-level modeling provides no explicit word boundaries or word-level semantic representation, so accumulated low-probability choices can create malformed words and destroy narrative structure.

### Testable Improvement

Use a lower temperature or nucleus sampling that removes the lowest-probability tail before sampling. Compare temperatures 0.7, 1.0, and 1.3 using grammatical-error counts, human coherence ratings, and diversity metrics. A useful decoding setting should preserve some diversity without the severe coherence loss observed here.

## Cross-Case Interpretation

The failures demonstrate a decoding tradeoff:

- Greedy and lower-entropy decoding can become repetitive.
- Moderate sampling can still enter repeated short-word patterns.
- High-temperature sampling increases diversity but damages grammar and coherence.

This interpretation is consistent with the measured results: temperature 1.3 achieved the highest Distinct-3 score (`0.339804`) and the lowest mean repeated 4-gram rate (`0.154706`), but its text was qualitatively less coherent. Therefore, diversity metrics should be interpreted alongside direct inspection of generated language.