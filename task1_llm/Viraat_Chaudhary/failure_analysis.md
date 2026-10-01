# Task 1 Failure Analysis - Viraat Chaudhary

Three manually reviewed failures from the saved model outputs.

## Case 1: Repetitive generation loop

Sample: `sample_001`; decoding: greedy; temperature: None.

> You are very sad. You are a good friend. You are very sad. You are a good friend. You are very sad. You are a good friend.

Observation: The same sentence pair appears three consecutive times in this excerpt, and the full continuation later repeats 'You are very sad.' many more times without developing the story. This greedy sample has a character-level repeated 4-gram rate of 0.748491. That is its individual rate; the reported aggregate rate of 0.174678 covers only the 27 sampled continuations and excludes greedy outputs.

Testable fix: As a decoding experiment, keep this checkpoint, the three prompts and the 500-character generation budget fixed. Compare greedy decoding with sampling at temperature 0.7 over several recorded seeds. Measure consecutive repeated-sentence runs and the character-level repeated 4-gram rate, and inspect grammar as well. Sampling may break the loop, but an improvement should be measured rather than assumed.

Evidence: outputs/samples/generated_samples.json

## Case 2: Unexplained speaker or character-name change

Sample: `sample_006`; decoding: sampled; temperature: 1.0.

> , there was a little girl called Jelie, "Zoomp! I love you. Thank you, Grandma!" Lily said.

Observation: The continuation introduces a girl called Jelie, but immediately attributes the quoted speech to Lily without introducing Lily or explaining the relationship. This leaves the speaker's identity ambiguous. The excerpt does not establish whether Lily is another character or an unintended replacement name. The example was sampled at temperature 1.0.

Testable fix: In a future training experiment, test whether additional training at a small learning rate improves character-name consistency while keeping the architecture and split fixed. Compare the resulting checkpoint with this frozen checkpoint on the same reserved prompts, generation length, temperature 1.0 and recorded sampling seeds. Count unexplained name changes and inspect validation loss. This is a proposed experiment, not a demonstrated fix.

Evidence: outputs/samples/generated_samples.json

## Case 3: Malformed words and grammar

Sample: `sample_018`; decoding: sampled; temperature: 1.3.

> Lily noticed down the tib legs. She splashed the stame. She's notice and kicked some chone flowersh.

Observation: The excerpt contains nonstandard word forms such as 'stame', 'chone' and 'flowersh', and 'She's notice' is grammatically malformed in this context. These errors make the events difficult to interpret even though the continuation contains familiar story words and a character name. This sample used temperature 1.3; one example cannot establish temperature as the only cause.

Testable fix: For a controlled decoding experiment, keep the checkpoint, prompts and 500-character generation budget fixed and compare temperatures 0.7, 1.0 and 1.3 over matched recorded seeds. Manually count malformed words and grammatical errors alongside repeated 4-gram rates. Test whether a lower temperature improves readability, and check whether that improvement comes with more repetition.

Evidence: outputs/samples/generated_samples.json
