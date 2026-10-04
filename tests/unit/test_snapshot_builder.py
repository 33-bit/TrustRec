import pytest
from scripts.build_snapshot import (
    is_software_game,
    split_interactions,
    text_duplicate_group,
    validate_review_row,
)


def test_entity_filter_keeps_games_and_rejects_hardware() -> None:
    assert is_software_game({"title": "A video game", "main_category": "Video Games"})
    assert not is_software_game(
        {"title": "Wireless game controller", "main_category": "Video Games"}
    )
    assert not is_software_game({"title": "Game console", "main_category": "Video Games"})


@pytest.mark.parametrize(
    "item_type",
    ("Video Game", "Software Download", "Game", "Computer Game"),
)
def test_entity_filter_keeps_allowed_item_types(item_type: str) -> None:
    assert is_software_game(
        {
            "title": "A product",
            "main_category": "Video Games",
            "details_json": {"Type of item": item_type},
        }
    )


@pytest.mark.parametrize(
    "item_type",
    ("Paperback", "Hardcover", "Accessory", "Console", "Electronics"),
)
def test_entity_filter_rejects_non_game_item_types(item_type: str) -> None:
    assert not is_software_game(
        {
            "title": "A product",
            "main_category": "Video Games",
            "details_json": {"Type of item": item_type},
        }
    )


def test_entity_filter_rejects_hardware_with_video_game_item_type() -> None:
    assert not is_software_game(
        {
            "title": "Wired Controller for Nintendo Switch",
            "main_category": "Video Games",
            "details": {"Type of item": "Video Game"},
        }
    )


def test_entity_filter_parses_serialized_details_json() -> None:
    assert is_software_game(
        {
            "title": "A product",
            "main_category": "Video Games",
            "details_json": '{"Type of item": "Software Download"}',
        }
    )


def test_entity_filter_rejects_book_with_game_word() -> None:
    assert not is_software_game(
        {
            "title": "The Game of Thrones novel",
            "main_category": "Books",
            "details_json": '{"Type of item": "Paperback"}',
        }
    )


def test_entity_filter_rejects_book_title_when_item_type_is_missing() -> None:
    assert not is_software_game(
        {
            "title": "A Coloring Book for Game Fans",
            "main_category": "Video Games",
            "details_json": "{}",
        }
    )


def test_entity_filter_accepts_raw_details_mapping() -> None:
    assert is_software_game(
        {
            "title": "A product",
            "main_category": "Video Games",
            "details": {"Type of item": "Video Game"},
        }
    )


def test_duplicate_group_is_item_scoped() -> None:
    assert text_duplicate_group("Same text", "i1") != text_duplicate_group("Same text", "i2")
    assert text_duplicate_group("Same text", "i1") == text_duplicate_group(" same  text ", "i1")


def test_split_uses_first_positive_target_for_repeated_pair() -> None:
    rows = [
        {"review_id": "late", "user_id": "u", "item_id": "i2", "rating": 5, "timestamp": 30},
        {"review_id": "first", "user_id": "u", "item_id": "i1", "rating": 5, "timestamp": 20},
        {"review_id": "repeat", "user_id": "u", "item_id": "i1", "rating": 5, "timestamp": 25},
        {"review_id": "history", "user_id": "u", "item_id": "i0", "rating": 5, "timestamp": 10},
        {"review_id": "catalog", "user_id": "v", "item_id": "i1", "rating": 3, "timestamp": 10},
    ]
    result = split_interactions(rows, 15, 30)
    assert [row["review_id"] for row in result["validation_targets"]] == ["first"]


def test_split_rejects_equal_cutoffs() -> None:
    with pytest.raises(ValueError, match="cutoff_t0"):
        split_interactions([], 10, 10)


def test_later_positive_rating_does_not_create_a_new_target() -> None:
    rows = [
        {"review_id": "catalog", "user_id": "v", "item_id": "i1", "rating": 5, "timestamp": 1},
        {"review_id": "first", "user_id": "u", "item_id": "i1", "rating": 2, "timestamp": 20},
        {"review_id": "repeat", "user_id": "u", "item_id": "i1", "rating": 5, "timestamp": 21},
    ]
    result = split_interactions(rows, 10, 30)
    assert result["validation_targets"] == []


def test_review_validation_requires_source_fields() -> None:
    with pytest.raises(ValueError, match="user_id"):
        validate_review_row("Video_Games", {"parent_asin": "i", "timestamp": 1})
