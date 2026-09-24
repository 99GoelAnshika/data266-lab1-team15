import torch

from data import (
    CharVocabulary,
    SequenceTensorDataset,
    build_sequence_tensor,
    collect_disjoint_story_text,
)


def test_vocabulary_round_trip() -> None:
    text = "a small story.\n"
    vocabulary = CharVocabulary.from_training_text(text)

    encoded = vocabulary.encode(text)
    decoded = vocabulary.decode(encoded.tolist())

    assert decoded == text
    assert vocabulary.unknown_index == 0
    assert vocabulary.idx_to_char[0] == "<UNK>"


def test_unknown_character_maps_to_unknown_index() -> None:
    vocabulary = CharVocabulary.from_training_text("abc")
    encoded = vocabulary.encode("az")

    assert encoded.tolist() == [
        vocabulary.char_to_idx["a"],
        vocabulary.unknown_index,
    ]


def test_input_target_shift() -> None:
    text = "abcdefghijkl"
    vocabulary = CharVocabulary.from_training_text(text)
    tensor = build_sequence_tensor(
        text=text,
        number_of_sequences=2,
        sequence_length=5,
        vocabulary=vocabulary,
    )
    dataset = SequenceTensorDataset(tensor)

    inputs, targets = dataset[0]

    assert tensor.shape == (2, 6)
    assert inputs.dtype == torch.long
    assert targets.dtype == torch.long
    assert vocabulary.decode(inputs) == "abcde"
    assert vocabulary.decode(targets) == "bcdef"
    assert torch.equal(inputs[1:], targets[:-1])


def test_story_groups_are_disjoint() -> None:
    dataset = [
        {"text": "abc"},
        {"text": "def"},
        {"text": "ghi"},
        {"text": "jkl"},
    ]

    train_text, next_index, train_story_count = (
        collect_disjoint_story_text(
            dataset=dataset,
            start_index=0,
            required_characters=5,
            text_field="text",
            separator="|",
        )
    )

    validation_text, final_index, validation_story_count = (
        collect_disjoint_story_text(
            dataset=dataset,
            start_index=next_index,
            required_characters=5,
            text_field="text",
            separator="|",
        )
    )

    assert train_text == "abc|d"
    assert validation_text == "ghi|j"
    assert next_index == 2
    assert final_index == 4
    assert train_story_count == 2
    assert validation_story_count == 2


def test_dataset_returns_fixed_length_pairs() -> None:
    vocabulary = CharVocabulary.from_training_text(
        "abcdefghijklmnop"
    )
    tensor = build_sequence_tensor(
        text="abcdefghijklmnop",
        number_of_sequences=2,
        sequence_length=7,
        vocabulary=vocabulary,
    )
    dataset = SequenceTensorDataset(tensor)

    assert len(dataset) == 2

    for inputs, targets in dataset:
        assert inputs.shape == (7,)
        assert targets.shape == (7,)
        assert torch.equal(inputs[1:], targets[:-1])