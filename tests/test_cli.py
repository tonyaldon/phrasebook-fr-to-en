import os
import contextlib
import threading
import time
from pathlib import Path
import logging
from typing import Any
from unittest.mock import Mock
import pytest
import pandas as pd
import phrasebook_fr_to_en.cli as cli
from typer.testing import CliRunner
import re
import httpx
from respx import MockRouter
from openai import OpenAI, APIError
from pydantic import TypeAdapter, conlist
from dotenv import load_dotenv
from mutagen.mp3 import MP3
from mutagen import MutagenError
from PIL import Image


if os.getenv("OPENAI_LIVE") == "1":
    load_dotenv()  # For OPENAI_API_KEY variable

runner = CliRunner()

## Utils


@contextlib.contextmanager
def disable_log_capture():
    from pytest import MonkeyPatch

    logger = logging.getLogger()
    with MonkeyPatch().context() as mp:
        mp.setattr(logger, "disabled", False)
        mp.setattr(logger, "handlers", [])
        mp.setattr(logger, "level", logging.NOTSET)
        yield


def is_mp3(path: Path) -> bool:
    try:
        MP3(path)
        return True
    except MutagenError:
        return False


def is_png(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            return im.format == "PNG"
    except Exception:
        return False


## Test `cli.enrich_record`


def test_enrich_record_ok(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mock_generate_translations = Mock(return_value=[("fr1", "en1"), ("fr2", "en2")])
    mock_generate_audio = Mock(return_value=None)
    mock_generate_img = Mock(return_value=None)
    monkeypatch.setattr(cli, "generate_translations", mock_generate_translations)
    monkeypatch.setattr(cli, "generate_audio", mock_generate_audio)
    monkeypatch.setattr(cli, "generate_img", mock_generate_img)

    record_original = ("2025-01-01", "bonjour", "hello")
    # We don't use client because generate function are mocked,
    # but we still have to pass as argument of `enrich_record`
    client = OpenAI(api_key="foo-api-key")
    phrasebook_dir = tmp_path  # We don't use it neither but must pass it as argument
    records = cli.enrich_record(record_original, 10, phrasebook_dir, client)

    assert mock_generate_audio.call_count == 3
    assert mock_generate_img.call_count == 3
    assert records == [
        {
            "id": 10,
            "french": "bonjour",
            "english": "hello",
            "anki_audio": "[sound:phrasebook-fr-to-en-10.mp3]",
            "anki_img": '<img src="phrasebook-fr-to-en-10.png">',
            "generated_from": pd.NA,
            "audio_filename": "phrasebook-fr-to-en-10.mp3",
            "img_filename": "phrasebook-fr-to-en-10.png",
            "date": "2025-01-01",
        },
        {
            "id": 11,
            "french": "fr1",
            "english": "en1",
            "anki_audio": "[sound:phrasebook-fr-to-en-11.mp3]",
            "anki_img": '<img src="phrasebook-fr-to-en-11.png">',
            "generated_from": 10,
            "audio_filename": "phrasebook-fr-to-en-11.mp3",
            "img_filename": "phrasebook-fr-to-en-11.png",
            "date": "2025-01-01",
        },
        {
            "id": 12,
            "french": "fr2",
            "english": "en2",
            "anki_audio": "[sound:phrasebook-fr-to-en-12.mp3]",
            "anki_img": '<img src="phrasebook-fr-to-en-12.png">',
            "generated_from": 10,
            "audio_filename": "phrasebook-fr-to-en-12.mp3",
            "img_filename": "phrasebook-fr-to-en-12.png",
            "date": "2025-01-01",
        },
    ]


def test_enrich_record_translation_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    mock_generate_translations = Mock(
        side_effect=RuntimeError("Failed in `generate_translations`")
    )
    mock_generate_audio = Mock(return_value=None)
    mock_generate_img = Mock(return_value=None)

    monkeypatch.setattr(cli, "generate_translations", mock_generate_translations)
    monkeypatch.setattr(cli, "generate_audio", mock_generate_audio)
    monkeypatch.setattr(cli, "generate_img", mock_generate_img)

    record_original = ("2025-01-01", "bonjour", "hello")
    # We don't use client because generate function are mocked,
    # but we still have to pass as argument of `enrich_record`
    client = OpenAI(api_key="foo-api-key")
    phrasebook_dir = tmp_path  # We don't use it neither but must pass it as argument
    records = cli.enrich_record(record_original, 10, phrasebook_dir, client)

    assert mock_generate_translations.called
    assert "Failed to generate translations while processing record" in caplog.text
    assert not mock_generate_audio.called
    assert not mock_generate_img.called
    assert records == []


def test_enrich_record_audio_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    mock_generate_translations = Mock(return_value=[("fr1", "en1"), ("fr2", "en2")])
    mock_generate_audio = Mock(side_effect=RuntimeError("Failed in `generate_audio`"))
    mock_generate_img = Mock(return_value=None)

    monkeypatch.setattr(cli, "generate_translations", mock_generate_translations)
    monkeypatch.setattr(cli, "generate_audio", mock_generate_audio)
    monkeypatch.setattr(cli, "generate_img", mock_generate_img)

    record_original = ("2025-01-01", "bonjour", "hello")
    # We don't use client because generate function are mocked,
    # but we still have to pass as argument of `enrich_record`
    client = OpenAI(api_key="foo-api-key")
    phrasebook_dir = tmp_path  # We don't use it neither but must pass it as argument
    records = cli.enrich_record(record_original, 10, phrasebook_dir, client)

    assert mock_generate_translations.called
    assert mock_generate_audio.called
    assert "Failed to generate audios while processing record" in caplog.text
    assert not mock_generate_img.called
    assert records == []


def test_enrich_record_img_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    mock_generate_translations = Mock(return_value=[("fr1", "en1"), ("fr2", "en2")])
    mock_generate_audio = Mock(return_value=None)
    mock_generate_img = Mock(side_effect=RuntimeError("Failed in `generate_img`"))

    monkeypatch.setattr(cli, "generate_translations", mock_generate_translations)
    monkeypatch.setattr(cli, "generate_audio", mock_generate_audio)
    monkeypatch.setattr(cli, "generate_img", mock_generate_img)

    record_original = ("2025-01-01", "bonjour", "hello")
    # We don't use client because generate function are mocked,
    # but we still have to pass as argument of `enrich_record`
    client = OpenAI(api_key="foo-api-key")
    phrasebook_dir = tmp_path  # We don't use it neither but must pass it as argument
    records = cli.enrich_record(record_original, 10, phrasebook_dir, client)

    assert mock_generate_translations.called
    assert mock_generate_audio.called
    assert mock_generate_img.called
    assert "Failed to generate images while processing record" in caplog.text
    assert records == []


## Test `cli.app`


def test_app_help():
    result = runner.invoke(cli.app, ["--help"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "Enrich French to English phrasebooks with OpenAI API." in result.output
    # help of `file` argument
    assert re.search(r"file.*Filename of the phrasebook to be enriched.", result.output)

    result = runner.invoke(cli.app, catch_exceptions=False)
    assert result.exit_code == 2, result.output
    assert "Missing argument 'FILE'." in result.output


def test_app_version() -> None:
    result = runner.invoke(cli.app, ["--version"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "phrasebook-fr-to-en " in result.output


def test_app_log_file_option(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # We exit with status 1 because OPENAI_API_KEY not set.
    # And we log the error to stderr (by default) because we don't
    # provide a --log-file.
    with disable_log_capture():
        result = runner.invoke(cli.app, ["some-filename"], catch_exceptions=False)

    assert result.exit_code == 1, result.output
    assert "Set OPENAI_API_KEY environment variable to run the app." in result.output

    # We exit with status 1 because OPENAI_API_KEY not set.
    # And we log the error in log_file file using --log-file option
    log_file = tmp_path / "logs"
    with disable_log_capture():
        result = runner.invoke(
            cli.app,
            ["--log-file", str(log_file), "some-filename"],
            catch_exceptions=False,
        )

    assert result.exit_code == 1, result.output
    assert (
        "Set OPENAI_API_KEY environment variable to run the app." not in result.output
    )
    assert (
        "Set OPENAI_API_KEY environment variable to run the app."
        in log_file.read_text()
    )


def test_app_errors(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    # OPENAI_API_KEY must be set to run the app
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = runner.invoke(cli.app, ["some-filename"], catch_exceptions=False)

    assert result.exit_code == 1, result.output
    assert "Set OPENAI_API_KEY environment variable to run the app." in caplog.text
    caplog.clear()

    # OPENAI_API_KEY must be set to run the app
    # As we mock `generate_...` functions, we don't hit OpenAI API,
    # so we don't have to use a real API key
    monkeypatch.setenv("OPENAI_API_KEY", "foo-api-key")

    # Phrasebook file doesn't exist
    tmp_path = tmp_path_factory.mktemp("phrasebook")
    phrasebook_path = tmp_path / "do-not-exist"

    result = runner.invoke(cli.app, [str(phrasebook_path)], catch_exceptions=False)

    assert result.exit_code == 1, result.output
    assert re.search(r"No such file or directory.*do-not-exist", caplog.text)
    caplog.clear()

    # Phrasebook file empty
    tmp_path = tmp_path_factory.mktemp("phrasebook")
    phrasebook_path = tmp_path / "phrasebook.tsv"
    phrasebook_path.touch()

    result = runner.invoke(cli.app, [str(phrasebook_path)], catch_exceptions=False)

    assert result.exit_code == 1, result.output
    assert "Invalid file" in caplog.text
    assert "No columns to parse from file" in caplog.text  # pandas error msg
    caplog.clear()

    # Phrasebook file exists but with wrong header fields
    tmp_path = tmp_path_factory.mktemp("phrasebook")
    phrasebook_path = tmp_path / "phrasebook.tsv"
    phrasebook_path.write_text("wrong_field_name\tfrench\tenglish")

    result = runner.invoke(cli.app, [str(phrasebook_path)], catch_exceptions=False)

    assert result.exit_code == 1, result.output
    assert "Invalid header" in caplog.text
    assert "Expected ['date', 'french', 'english']" in caplog.text
    assert "got ['wrong_field_name', 'french', 'english']" in caplog.text
    caplog.clear()

    # enriched_phrasebook.tsv file empty
    tmp_path = tmp_path_factory.mktemp("phrasebook")
    phrasebook_path = tmp_path / "phrasebook.tsv"
    phrasebook_path.write_text("date\tfrench\tenglish\n2025-12-15\tfr-foo\ten-foo")
    enriched_path = cli.enriched_path_func(phrasebook_path)
    enriched_path.touch()

    result = runner.invoke(cli.app, [str(phrasebook_path)], catch_exceptions=False)

    assert result.exit_code == 1, result.output
    assert "Invalid file" in caplog.text
    assert "No columns to parse from file" in caplog.text  # pandas error msg
    caplog.clear()

    # enriched_phrasebook.tsv exists but with wrong header fields
    tmp_path = tmp_path_factory.mktemp("phrasebook")
    phrasebook_path = tmp_path / "phrasebook.tsv"
    phrasebook_path.write_text("date\tfrench\tenglish\n2025-12-15\tfr-foo\ten-foo")
    enriched_path = cli.enriched_path_func(phrasebook_path)
    enriched_path.write_text("foo\tbar\tbaz")

    result = runner.invoke(cli.app, [str(phrasebook_path)], catch_exceptions=False)

    assert result.exit_code == 1, result.output
    assert "Invalid header" in caplog.text
    assert (
        "Expected ['french', 'english', 'anki_audio', 'anki_img', 'generated_from', 'id', 'audio_filename', 'img_filename', 'date']"
        in caplog.text
    )
    assert "got ['foo', 'bar', 'baz']" in caplog.text
    caplog.clear()

    # `enrich_record` failed in some way so returns None and we exit
    tmp_path = tmp_path_factory.mktemp("phrasebook")
    phrasebook_path = tmp_path / "phrasebook.tsv"
    phrasebook_path.write_text("date\tfrench\tenglish\n2025-12-15\tfr-foo\ten-foo")
    mock_enrich_record = Mock(return_value=None)
    monkeypatch.setattr(cli, "enrich_record", mock_enrich_record)

    result = runner.invoke(cli.app, [str(phrasebook_path)], catch_exceptions=False)

    assert result.exit_code == 1, result.output
    assert mock_enrich_record.called
    caplog.clear()

    # `save_new_records` raising an error we exit
    tmp_path = tmp_path_factory.mktemp("phrasebook")
    phrasebook_path = tmp_path / "phrasebook.tsv"
    phrasebook_path.write_text("date\tfrench\tenglish\n2025-12-15\tfr-foo\ten-foo")
    mock_enrich_record = Mock(
        return_value=True
    )  # To continue in `cli.enrich_phrasebook` function.  Not that in real this should be list of records
    mock_save_new_records = Mock(
        side_effect=RuntimeError("Failed in `save_new_records`")
    )
    monkeypatch.setattr(cli, "enrich_record", mock_enrich_record)
    monkeypatch.setattr(cli, "save_new_records", mock_save_new_records)

    result = runner.invoke(cli.app, [str(phrasebook_path)], catch_exceptions=False)

    assert mock_enrich_record.called
    assert mock_save_new_records.called
    assert result.exit_code == 1, result.output
    assert "Failed to save enriched records from record" in caplog.text


@pytest.mark.parametrize(
    "phrasebook_content,translations,enriched_content,enriched_expected,logs",
    [
        # 1 record to be enriched + enriched_phrasebook.tsv doesn't exist
        (
            "date\tfrench\tenglish\n2025-12-15\tfr1\ten1",
            [[("fr2", "en2"), ("fr3", "en3")]],
            None,
            [
                ("fr1", "en1", "[sound:phrasebook-fr-to-en-1.mp3]", "<img src=\"phrasebook-fr-to-en-1.png\">", pd.NA, 1, "phrasebook-fr-to-en-1.mp3", "phrasebook-fr-to-en-1.png", "2025-12-15"),
                ("fr2", "en2", "[sound:phrasebook-fr-to-en-2.mp3]", "<img src=\"phrasebook-fr-to-en-2.png\">", 1,     2, "phrasebook-fr-to-en-2.mp3", "phrasebook-fr-to-en-2.png", "2025-12-15"),
                ("fr3", "en3", "[sound:phrasebook-fr-to-en-3.mp3]", "<img src=\"phrasebook-fr-to-en-3.png\">", 1,     3, "phrasebook-fr-to-en-3.mp3", "phrasebook-fr-to-en-3.png", "2025-12-15"),
            ],
            ["Record has been enriched: ('2025-12-15', 'fr1', 'en1')"],
        ),
        # 1 record / enriched file 3 records (corresponding to 1 record from a previous run)
        # english NOT in phrasebook_content
        # should create 3 new enriched records and keep the original 3
        (
            "date\tfrench\tenglish\n2025-12-15\tfr4\ten4",
            [[("fr5", "en5"), ("fr6", "en6")]],
            (
                "french\tenglish\tanki_audio\tanki_img\tgenerated_from\tid\taudio_filename\timg_filename\tdate\n"
                'fr1\ten1\t[sound:phrasebook-fr-to-en-1.mp3]\t"<img src=""phrasebook-fr-to-en-1.png"">"\t\t1\tphrasebook-fr-to-en-1.mp3\tphrasebook-fr-to-en-1.png\t2025-12-01\n'
                'fr2\ten2\t[sound:phrasebook-fr-to-en-2.mp3]\t"<img src=""phrasebook-fr-to-en-2.png"">"\t1\t2\tphrasebook-fr-to-en-2.mp3\tphrasebook-fr-to-en-2.png\t2025-12-01\n'
                'fr3\ten3\t[sound:phrasebook-fr-to-en-3.mp3]\t"<img src=""phrasebook-fr-to-en-3.png"">"\t1\t3\tphrasebook-fr-to-en-3.mp3\tphrasebook-fr-to-en-3.png\t2025-12-01'
            ),
            [
                ("fr1", "en1", "[sound:phrasebook-fr-to-en-1.mp3]", "<img src=\"phrasebook-fr-to-en-1.png\">", pd.NA,  1, "phrasebook-fr-to-en-1.mp3", "phrasebook-fr-to-en-1.png", "2025-12-01",),
                ("fr2", "en2", "[sound:phrasebook-fr-to-en-2.mp3]", "<img src=\"phrasebook-fr-to-en-2.png\">", 1,      2, "phrasebook-fr-to-en-2.mp3", "phrasebook-fr-to-en-2.png", "2025-12-01"),
                ("fr3", "en3", "[sound:phrasebook-fr-to-en-3.mp3]", "<img src=\"phrasebook-fr-to-en-3.png\">", 1,      3, "phrasebook-fr-to-en-3.mp3", "phrasebook-fr-to-en-3.png", "2025-12-01"),
                ("fr4", "en4", "[sound:phrasebook-fr-to-en-4.mp3]", "<img src=\"phrasebook-fr-to-en-4.png\">", pd.NA,  4, "phrasebook-fr-to-en-4.mp3", "phrasebook-fr-to-en-4.png", "2025-12-15",),
                ("fr5", "en5", "[sound:phrasebook-fr-to-en-5.mp3]", "<img src=\"phrasebook-fr-to-en-5.png\">", 4,      5, "phrasebook-fr-to-en-5.mp3", "phrasebook-fr-to-en-5.png", "2025-12-15"),
                ("fr6", "en6", "[sound:phrasebook-fr-to-en-6.mp3]", "<img src=\"phrasebook-fr-to-en-6.png\">", 4,      6, "phrasebook-fr-to-en-6.mp3", "phrasebook-fr-to-en-6.png", "2025-12-15"),
            ],
            ["Record has been enriched: ('2025-12-15', 'fr4', 'en4')"],
        ),
        # 1 record / enriched file 3 records (corresponding to 1 record from a previous run)
        # 'en1' english field is present in both phrasebook_content and enriched_content
        # should not create new enriched records
        # "Skip..." in the logs
        (
            "date\tfrench\tenglish\n2025-12-15\tfr_whatever\ten1",
            None, # No translations generated
            (
                "french\tenglish\tanki_audio\tanki_img\tgenerated_from\tid\taudio_filename\timg_filename\tdate\n"
                'fr1\ten1\t[sound:phrasebook-fr-to-en-1.mp3]\t"<img src=""phrasebook-fr-to-en-1.png"">"\t\t1\tphrasebook-fr-to-en-1.mp3\tphrasebook-fr-to-en-1.png\t2025-12-01\n'
                'fr2\ten2\t[sound:phrasebook-fr-to-en-2.mp3]\t"<img src=""phrasebook-fr-to-en-2.png"">"\t1\t2\tphrasebook-fr-to-en-2.mp3\tphrasebook-fr-to-en-2.png\t2025-12-01\n'
                'fr3\ten3\t[sound:phrasebook-fr-to-en-3.mp3]\t"<img src=""phrasebook-fr-to-en-3.png"">"\t1\t3\tphrasebook-fr-to-en-3.mp3\tphrasebook-fr-to-en-3.png\t2025-12-01'
            ),
            [
                ("fr1", "en1", "[sound:phrasebook-fr-to-en-1.mp3]", "<img src=\"phrasebook-fr-to-en-1.png\">", pd.NA, 1, "phrasebook-fr-to-en-1.mp3", "phrasebook-fr-to-en-1.png", "2025-12-01"),
                ("fr2", "en2", "[sound:phrasebook-fr-to-en-2.mp3]", "<img src=\"phrasebook-fr-to-en-2.png\">", 1,     2, "phrasebook-fr-to-en-2.mp3", "phrasebook-fr-to-en-2.png", "2025-12-01"),
                ("fr3", "en3", "[sound:phrasebook-fr-to-en-3.mp3]", "<img src=\"phrasebook-fr-to-en-3.png\">", 1,     3, "phrasebook-fr-to-en-3.mp3", "phrasebook-fr-to-en-3.png", "2025-12-01"),
            ],
            ["Skip existing record: ('2025-12-15', 'fr_whatever', 'en1')"],
        ),
        # phrasebook_content 3 records / no enriched file
        # should create 9 enriched records
        (
            "date\tfrench\tenglish\n"
            "2025-12-15\tfr1\ten1\n"
            "2025-12-16\tfr4\ten4\n"
            "2025-12-17\tfr7\ten7",
            [
                [("fr2", "en2"), ("fr3", "en3")],
                [("fr5", "en5"), ("fr6", "en6")],
                [("fr8", "en8"), ("fr9", "en9")],
            ],
            None,
            [
                ("fr1", "en1", "[sound:phrasebook-fr-to-en-1.mp3]", "<img src=\"phrasebook-fr-to-en-1.png\">", pd.NA, 1, "phrasebook-fr-to-en-1.mp3", "phrasebook-fr-to-en-1.png", "2025-12-15"),
                ("fr2", "en2", "[sound:phrasebook-fr-to-en-2.mp3]", "<img src=\"phrasebook-fr-to-en-2.png\">", 1,     2, "phrasebook-fr-to-en-2.mp3", "phrasebook-fr-to-en-2.png", "2025-12-15"),
                ("fr3", "en3", "[sound:phrasebook-fr-to-en-3.mp3]", "<img src=\"phrasebook-fr-to-en-3.png\">", 1,     3, "phrasebook-fr-to-en-3.mp3", "phrasebook-fr-to-en-3.png", "2025-12-15"),
                ("fr4", "en4", "[sound:phrasebook-fr-to-en-4.mp3]", "<img src=\"phrasebook-fr-to-en-4.png\">", pd.NA, 4, "phrasebook-fr-to-en-4.mp3", "phrasebook-fr-to-en-4.png", "2025-12-16"),
                ("fr5", "en5", "[sound:phrasebook-fr-to-en-5.mp3]", "<img src=\"phrasebook-fr-to-en-5.png\">", 4,     5, "phrasebook-fr-to-en-5.mp3", "phrasebook-fr-to-en-5.png", "2025-12-16"),
                ("fr6", "en6", "[sound:phrasebook-fr-to-en-6.mp3]", "<img src=\"phrasebook-fr-to-en-6.png\">", 4,     6, "phrasebook-fr-to-en-6.mp3", "phrasebook-fr-to-en-6.png", "2025-12-16"),
                ("fr7", "en7", "[sound:phrasebook-fr-to-en-7.mp3]", "<img src=\"phrasebook-fr-to-en-7.png\">", pd.NA, 7, "phrasebook-fr-to-en-7.mp3", "phrasebook-fr-to-en-7.png", "2025-12-17"),
                ("fr8", "en8", "[sound:phrasebook-fr-to-en-8.mp3]", "<img src=\"phrasebook-fr-to-en-8.png\">", 7,     8, "phrasebook-fr-to-en-8.mp3", "phrasebook-fr-to-en-8.png", "2025-12-17"),
                ("fr9", "en9", "[sound:phrasebook-fr-to-en-9.mp3]", "<img src=\"phrasebook-fr-to-en-9.png\">", 7,     9, "phrasebook-fr-to-en-9.mp3", "phrasebook-fr-to-en-9.png", "2025-12-17"),
            ],
            [
                "Record has been enriched: ('2025-12-15', 'fr1', 'en1')",
                "Record has been enriched: ('2025-12-16', 'fr4', 'en4')",
                "Record has been enriched: ('2025-12-17', 'fr7', 'en7')",
            ],
        ),
        # phrasebook_content 3 records / enriched file 6 records (corresponding to 2 records from a previous run)
        # english NOT in phrasebook_content
        # should create 9 enriched records and keep the original 6
        (
            "date\tfrench\tenglish\n"
            "2025-12-15\tfr7\ten7\n"
            "2025-12-16\tfr10\ten10\n"
            "2025-12-17\tfr13\ten13",
            [
                [("fr8", "en8"), ("fr9", "en9")],
                [("fr11", "en11"), ("fr12", "en12")],
                [("fr14", "en14"), ("fr15", "en15")],
            ],
            (
                "french\tenglish\tanki_audio\tanki_img\tgenerated_from\tid\taudio_filename\timg_filename\tdate\n"
                'fr1\ten1\t[sound:phrasebook-fr-to-en-1.mp3]\t"<img src=""phrasebook-fr-to-en-1.png"">"\t\t1\tphrasebook-fr-to-en-1.mp3\tphrasebook-fr-to-en-1.png\t2025-12-01\n'
                'fr2\ten2\t[sound:phrasebook-fr-to-en-2.mp3]\t"<img src=""phrasebook-fr-to-en-2.png"">"\t1\t2\tphrasebook-fr-to-en-2.mp3\tphrasebook-fr-to-en-2.png\t2025-12-01\n'
                'fr3\ten3\t[sound:phrasebook-fr-to-en-3.mp3]\t"<img src=""phrasebook-fr-to-en-3.png"">"\t1\t3\tphrasebook-fr-to-en-3.mp3\tphrasebook-fr-to-en-3.png\t2025-12-01\n'
                'fr4\ten4\t[sound:phrasebook-fr-to-en-4.mp3]\t"<img src=""phrasebook-fr-to-en-4.png"">"\t\t4\tphrasebook-fr-to-en-4.mp3\tphrasebook-fr-to-en-4.png\t2025-12-02\n'
                'fr5\ten5\t[sound:phrasebook-fr-to-en-5.mp3]\t"<img src=""phrasebook-fr-to-en-5.png"">"\t4\t5\tphrasebook-fr-to-en-5.mp3\tphrasebook-fr-to-en-5.png\t2025-12-02\n'
                'fr6\ten6\t[sound:phrasebook-fr-to-en-6.mp3]\t"<img src=""phrasebook-fr-to-en-6.png"">"\t4\t6\tphrasebook-fr-to-en-6.mp3\tphrasebook-fr-to-en-6.png\t2025-12-02'
            ),
            [
                ("fr1",  "en1",  "[sound:phrasebook-fr-to-en-1.mp3]",  "<img src=\"phrasebook-fr-to-en-1.png\">",  pd.NA, 1,  "phrasebook-fr-to-en-1.mp3", "phrasebook-fr-to-en-1.png", "2025-12-01",),
                ("fr2",  "en2",  "[sound:phrasebook-fr-to-en-2.mp3]",  "<img src=\"phrasebook-fr-to-en-2.png\">",  1,     2,  "phrasebook-fr-to-en-2.mp3", "phrasebook-fr-to-en-2.png", "2025-12-01",),
                ("fr3",  "en3",  "[sound:phrasebook-fr-to-en-3.mp3]",  "<img src=\"phrasebook-fr-to-en-3.png\">",  1,     3,  "phrasebook-fr-to-en-3.mp3", "phrasebook-fr-to-en-3.png", "2025-12-01",),
                ("fr4",  "en4",  "[sound:phrasebook-fr-to-en-4.mp3]",  "<img src=\"phrasebook-fr-to-en-4.png\">",  pd.NA, 4,  "phrasebook-fr-to-en-4.mp3", "phrasebook-fr-to-en-4.png", "2025-12-02",),
                ("fr5",  "en5",  "[sound:phrasebook-fr-to-en-5.mp3]",  "<img src=\"phrasebook-fr-to-en-5.png\">",  4,     5,  "phrasebook-fr-to-en-5.mp3", "phrasebook-fr-to-en-5.png", "2025-12-02",),
                ("fr6",  "en6",  "[sound:phrasebook-fr-to-en-6.mp3]",  "<img src=\"phrasebook-fr-to-en-6.png\">",  4,     6,  "phrasebook-fr-to-en-6.mp3", "phrasebook-fr-to-en-6.png", "2025-12-02",),
                ("fr7",  "en7",  "[sound:phrasebook-fr-to-en-7.mp3]",  "<img src=\"phrasebook-fr-to-en-7.png\">",  pd.NA, 7,  "phrasebook-fr-to-en-7.mp3", "phrasebook-fr-to-en-7.png", "2025-12-15"),
                ("fr8",  "en8",  "[sound:phrasebook-fr-to-en-8.mp3]",  "<img src=\"phrasebook-fr-to-en-8.png\">",  7,     8,  "phrasebook-fr-to-en-8.mp3", "phrasebook-fr-to-en-8.png", "2025-12-15"),
                ("fr9",  "en9",  "[sound:phrasebook-fr-to-en-9.mp3]",  "<img src=\"phrasebook-fr-to-en-9.png\">",  7,     9,  "phrasebook-fr-to-en-9.mp3", "phrasebook-fr-to-en-9.png", "2025-12-15"),
                ("fr10", "en10", "[sound:phrasebook-fr-to-en-10.mp3]", "<img src=\"phrasebook-fr-to-en-10.png\">", pd.NA, 10, "phrasebook-fr-to-en-10.mp3", "phrasebook-fr-to-en-10.png", "2025-12-16"),
                ("fr11", "en11", "[sound:phrasebook-fr-to-en-11.mp3]", "<img src=\"phrasebook-fr-to-en-11.png\">", 10,    11, "phrasebook-fr-to-en-11.mp3", "phrasebook-fr-to-en-11.png", "2025-12-16"),
                ("fr12", "en12", "[sound:phrasebook-fr-to-en-12.mp3]", "<img src=\"phrasebook-fr-to-en-12.png\">", 10,    12, "phrasebook-fr-to-en-12.mp3", "phrasebook-fr-to-en-12.png", "2025-12-16"),
                ("fr13", "en13", "[sound:phrasebook-fr-to-en-13.mp3]", "<img src=\"phrasebook-fr-to-en-13.png\">", pd.NA, 13, "phrasebook-fr-to-en-13.mp3", "phrasebook-fr-to-en-13.png", "2025-12-17"),
                ("fr14", "en14", "[sound:phrasebook-fr-to-en-14.mp3]", "<img src=\"phrasebook-fr-to-en-14.png\">", 13,    14, "phrasebook-fr-to-en-14.mp3", "phrasebook-fr-to-en-14.png", "2025-12-17"),
                ("fr15", "en15", "[sound:phrasebook-fr-to-en-15.mp3]", "<img src=\"phrasebook-fr-to-en-15.png\">", 13,    15, "phrasebook-fr-to-en-15.mp3", "phrasebook-fr-to-en-15.png", "2025-12-17"),
            ],
            [
                "Record has been enriched: ('2025-12-15', 'fr7', 'en7')",
                "Record has been enriched: ('2025-12-16', 'fr10', 'en10')",
                "Record has been enriched: ('2025-12-17', 'fr13', 'en13')",
            ],
        ),
        # phrasebook_content 3 records / enriched file 6 records (corresponding to 2 records from a previous run)
        # 'en1' english field is present in both phrasebook_content and enriched_content
        # should create only 6 new enriched records for the other 2 phrasebook records
        # "Skip..." in the logs
        (
            "date\tfrench\tenglish\n"
            "2025-12-15\tfr_whatever\ten1\n"
            "2025-12-16\tfr7\ten7\n"
            "2025-12-17\tfr10\ten10",
            [
                [("fr8", "en8"), ("fr9", "en9")],
                [("fr11", "en11"), ("fr12", "en12")],

            ],
            (
                "french\tenglish\tanki_audio\tanki_img\tgenerated_from\tid\taudio_filename\timg_filename\tdate\n"
                'fr1\ten1\t[sound:phrasebook-fr-to-en-1.mp3]\t"<img src=""phrasebook-fr-to-en-1.png"">"\t\t1\tphrasebook-fr-to-en-1.mp3\tphrasebook-fr-to-en-1.png\t2025-12-01\n'
                'fr2\ten2\t[sound:phrasebook-fr-to-en-2.mp3]\t"<img src=""phrasebook-fr-to-en-2.png"">"\t1\t2\tphrasebook-fr-to-en-2.mp3\tphrasebook-fr-to-en-2.png\t2025-12-01\n'
                'fr3\ten3\t[sound:phrasebook-fr-to-en-3.mp3]\t"<img src=""phrasebook-fr-to-en-3.png"">"\t1\t3\tphrasebook-fr-to-en-3.mp3\tphrasebook-fr-to-en-3.png\t2025-12-01\n'
                'fr4\ten4\t[sound:phrasebook-fr-to-en-4.mp3]\t"<img src=""phrasebook-fr-to-en-4.png"">"\t\t4\tphrasebook-fr-to-en-4.mp3\tphrasebook-fr-to-en-4.png\t2025-12-02\n'
                'fr5\ten5\t[sound:phrasebook-fr-to-en-5.mp3]\t"<img src=""phrasebook-fr-to-en-5.png"">"\t4\t5\tphrasebook-fr-to-en-5.mp3\tphrasebook-fr-to-en-5.png\t2025-12-02\n'
                'fr6\ten6\t[sound:phrasebook-fr-to-en-6.mp3]\t"<img src=""phrasebook-fr-to-en-6.png"">"\t4\t6\tphrasebook-fr-to-en-6.mp3\tphrasebook-fr-to-en-6.png\t2025-12-02'
            ),
            [
                ("fr1",  "en1",  "[sound:phrasebook-fr-to-en-1.mp3]",  "<img src=\"phrasebook-fr-to-en-1.png\">",  pd.NA, 1,  "phrasebook-fr-to-en-1.mp3", "phrasebook-fr-to-en-1.png", "2025-12-01",),
                ("fr2",  "en2",  "[sound:phrasebook-fr-to-en-2.mp3]",  "<img src=\"phrasebook-fr-to-en-2.png\">",  1,     2,  "phrasebook-fr-to-en-2.mp3", "phrasebook-fr-to-en-2.png", "2025-12-01",),
                ("fr3",  "en3",  "[sound:phrasebook-fr-to-en-3.mp3]",  "<img src=\"phrasebook-fr-to-en-3.png\">",  1,     3,  "phrasebook-fr-to-en-3.mp3", "phrasebook-fr-to-en-3.png", "2025-12-01",),
                ("fr4",  "en4",  "[sound:phrasebook-fr-to-en-4.mp3]",  "<img src=\"phrasebook-fr-to-en-4.png\">",  pd.NA, 4,  "phrasebook-fr-to-en-4.mp3", "phrasebook-fr-to-en-4.png", "2025-12-02",),
                ("fr5",  "en5",  "[sound:phrasebook-fr-to-en-5.mp3]",  "<img src=\"phrasebook-fr-to-en-5.png\">",  4,     5,  "phrasebook-fr-to-en-5.mp3", "phrasebook-fr-to-en-5.png", "2025-12-02",),
                ("fr6",  "en6",  "[sound:phrasebook-fr-to-en-6.mp3]",  "<img src=\"phrasebook-fr-to-en-6.png\">",  4,     6,  "phrasebook-fr-to-en-6.mp3", "phrasebook-fr-to-en-6.png", "2025-12-02",),
                ("fr7",  "en7",  "[sound:phrasebook-fr-to-en-7.mp3]",  "<img src=\"phrasebook-fr-to-en-7.png\">",  pd.NA, 7,  "phrasebook-fr-to-en-7.mp3", "phrasebook-fr-to-en-7.png", "2025-12-16"),
                ("fr8",  "en8",  "[sound:phrasebook-fr-to-en-8.mp3]",  "<img src=\"phrasebook-fr-to-en-8.png\">",  7,     8,  "phrasebook-fr-to-en-8.mp3", "phrasebook-fr-to-en-8.png", "2025-12-16"),
                ("fr9",  "en9",  "[sound:phrasebook-fr-to-en-9.mp3]",  "<img src=\"phrasebook-fr-to-en-9.png\">",  7,     9,  "phrasebook-fr-to-en-9.mp3", "phrasebook-fr-to-en-9.png", "2025-12-16"),
                ("fr10", "en10", "[sound:phrasebook-fr-to-en-10.mp3]", "<img src=\"phrasebook-fr-to-en-10.png\">", pd.NA, 10, "phrasebook-fr-to-en-10.mp3", "phrasebook-fr-to-en-10.png", "2025-12-17"),
                ("fr11", "en11", "[sound:phrasebook-fr-to-en-11.mp3]", "<img src=\"phrasebook-fr-to-en-11.png\">", 10,    11, "phrasebook-fr-to-en-11.mp3", "phrasebook-fr-to-en-11.png", "2025-12-17"),
                ("fr12", "en12", "[sound:phrasebook-fr-to-en-12.mp3]", "<img src=\"phrasebook-fr-to-en-12.png\">", 10,    12, "phrasebook-fr-to-en-12.mp3", "phrasebook-fr-to-en-12.png", "2025-12-17"),
            ],
            [
                "Skip existing record: ('2025-12-15', 'fr_whatever', 'en1')",
                "Record has been enriched: ('2025-12-16', 'fr7', 'en7')",
                "Record has been enriched: ('2025-12-17', 'fr10', 'en10')",
            ],
        ),
    ],
    ids=[
        "1_record_no_enriched_file",
        "1_record_enriched_file_3_records_english_not_in_phrasebook_creates_3_keeps_3",
        "1_record_enriched_file_3_records_english_same_skips_creates_0",
        "3_records_no_enriched_file_creates_9",
        "3_records_enriched_file_6_records_english_not_in_phrasebook_creates_9_keeps_6",
        "3_records_enriched_file_6_records_first_english_same_skips_1_creates_6",
    ],
)  # fmt: skip
def test_app_records_saved(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    phrasebook_content,
    translations,
    enriched_content,
    enriched_expected,
    logs,
):
    caplog.set_level(logging.INFO, logger="phrasebook_fr_to_en.cli")

    # OPENAI_API_KEY must be set to run the app
    # As we mock `generate_...` functions, we don't hit OpenAI API,
    # so we don't have to use a real API key
    monkeypatch.setenv("OPENAI_API_KEY", "foo-api-key")

    mock_generate_translations = Mock(side_effect=translations)
    mock_generate_audio = Mock(return_value=None)
    mock_generate_img = Mock(return_value=None)
    monkeypatch.setattr(cli, "generate_translations", mock_generate_translations)
    monkeypatch.setattr(cli, "generate_audio", mock_generate_audio)
    monkeypatch.setattr(cli, "generate_img", mock_generate_img)

    tmp_path = tmp_path_factory.mktemp("phrasebook")
    phrasebook_path = tmp_path / "phrasebook.tsv"
    phrasebook_path.write_text(phrasebook_content)
    enriched_path = cli.enriched_path_func(phrasebook_path)
    if enriched_content:
        enriched_path.write_text(enriched_content)

    result = runner.invoke(cli.app, [str(phrasebook_path)], catch_exceptions=False)

    assert result.exit_code == 0, result.output

    enriched_df = pd.read_csv(enriched_path, sep="\t", dtype="string")
    # Match the dtypes produced by save_new_records
    enriched_df["id"] = enriched_df["id"].astype("Int64")
    enriched_df["generated_from"] = enriched_df["generated_from"].astype("Int64")
    enriched_df_expected = pd.DataFrame(
        enriched_expected,
        columns=pd.Index(cli.ENRICHED_COLUMNS),
        dtype="string",
    )
    enriched_df_expected["id"] = enriched_df_expected["id"].astype("Int64")
    enriched_df_expected["generated_from"] = enriched_df_expected[
        "generated_from"
    ].astype("Int64")

    pd.testing.assert_frame_equal(enriched_df, enriched_df_expected, check_dtype=True)

    for log in logs:
        assert log in caplog.text


## Test `cli.watch_phrasebook`


def test_watch_phrasebook(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="phrasebook_fr_to_en.cli")

    translations: list[list[tuple[str, str]]] = [
        [("fr1-a", "en1-a"), ("fr1-b", "en1-b")],
        [("fr2-a", "en2-a"), ("fr2-b", "en2-b")],
        [("fr3-a", "en3-a"), ("fr3-b", "en3-b")],
    ]
    mock_generate_translations = Mock(side_effect=translations)
    mock_generate_audio = Mock(return_value=None)
    mock_generate_img = Mock(return_value=None)
    monkeypatch.setattr(cli, "generate_translations", mock_generate_translations)
    monkeypatch.setattr(cli, "generate_audio", mock_generate_audio)
    monkeypatch.setattr(cli, "generate_img", mock_generate_img)

    tmp_path = tmp_path_factory.mktemp("phrasebook")
    phrasebook_path = tmp_path / "phrasebook.tsv"
    phrasebook_path.write_text("date\tfrench\tenglish\n2025-12-15\tfr1\ten1")
    enriched_path = cli.enriched_path_func(phrasebook_path)

    # We don't use client because generate function are mocked,
    # but we still have to pass as argument of `watch_phrasebook`
    client = OpenAI(api_key="foo-api-key")

    # We start watching
    # `enriched_path` file doesn't exist
    thread = threading.Thread(
        target=cli.watch_phrasebook,
        args=[phrasebook_path.absolute(), client],
        daemon=True,
    )
    thread.start()

    deadline = time.monotonic() + 5.0
    while (
        time.monotonic() < deadline
        and f"Start watching file {phrasebook_path}" not in caplog.text
    ):
        time.sleep(0.01)

    assert f"Start watching file {phrasebook_path}" in caplog.text
    caplog.clear()

    def read_enriched_english_values() -> list[str]:
        # While `enriched_path` is being written or before it has been
        # created, `pd.read_csv()` raises an error
        try:
            enriched_df = pd.read_csv(enriched_path, sep="\t", dtype="string")
            return enriched_df["english"].dropna().to_list()
        except Exception:
            return []

    # enrich "en1" and "en2" records
    phrasebook_path.write_text(
        "date\tfrench\tenglish\n2025-12-15\tfr1\ten1\n2025-12-16\tfr2\ten2"
    )

    deadline = time.monotonic() + 5.0
    while (
        time.monotonic() < deadline
        and "en1" not in read_enriched_english_values()
        and "en2" not in read_enriched_english_values()
    ):
        time.sleep(0.01)

    assert "Skip existing record: ('2025-12-15', 'fr1', 'en1')" not in caplog.text
    assert "en1" in read_enriched_english_values()
    assert "en2" in read_enriched_english_values()
    caplog.clear()

    # Invalid file logged due to last row having 4 fields instead of 3
    phrasebook_path.write_text(
        "date\tfrench\tenglish\n"
        "2025-12-15\tfr1\ten1\n"
        "2025-12-16\tfr2\ten2\n"
        "2025-12-17\tfr3\ten3\twrong-extra-field"
    )

    deadline = time.monotonic() + 5.0
    while (
        time.monotonic() < deadline
        and "Invalid file" not in caplog.text
        and "Expected 3 fields in line 4, saw 4" not in caplog.text
    ):
        time.sleep(0.01)

    assert f"Invalid file {phrasebook_path}" in caplog.text
    assert "Expected 3 fields in line 4, saw 4" in caplog.text  # pandas error msg
    caplog.clear()

    # Skip existing "en1" and "en2" records and enriched "en2" record
    phrasebook_path.write_text(
        "date\tfrench\tenglish\n"
        "2025-12-15\tfr1\ten1\n"
        "2025-12-16\tfr2\ten2\n"
        "2025-12-17\tfr3\ten3"
    )

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and "en3" not in read_enriched_english_values():
        time.sleep(0.01)

    assert "Skip existing record: ('2025-12-15', 'fr1', 'en1')" in caplog.text
    assert "Skip existing record: ('2025-12-16', 'fr2', 'en2')" in caplog.text
    assert "en3" in read_enriched_english_values()

    assert thread.is_alive()


@pytest.mark.respx(base_url="https://api.openai.com/v1/")
def test_generate_translations(
    respx_mock: MockRouter, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.INFO, logger="phrasebook_fr_to_en.cli")

    record = ("2025-12-15", "fr1", "en1")
    client = OpenAI(api_key="foo-api-key")

    def partial_json_response(output_id: str, output_text: str):
        return {
            "output": [
                {
                    "type": "message",
                    "id": f"msg_{output_id}",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": output_text,
                            "annotations": [],
                        }
                    ],
                }
            ]
        }

    # We receive exactly 2 translations and this is what we want
    respx_mock.post("/responses").mock(
        return_value=httpx.Response(
            200,
            json=partial_json_response(
                "id_1",
                '{"translations":[{"french":"fr2","english":"en2"},{"french":"fr3","english":"en3"}]}',
            ),
        )
    )

    translations = cli.generate_translations(record, client)

    assert translations == [("fr2", "en2"), ("fr3", "en3")]
    assert (
        "Generating translations for record ('2025-12-15', 'fr1', 'en1')" in caplog.text
    )
    assert (
        "Translations generated for record ('2025-12-15', 'fr1', 'en1')" in caplog.text
    )
    assert "using model gpt-5.2 and input 'fr1 -> en1'" in caplog.text
    caplog.clear()

    # First request returns 3 translations -> We take the first 2
    # Second request would return 2 translations, which is ok, but we
    # never send that second request because we stopped at the first one.
    respx_mock.post("/responses").mock(
        side_effect=[
            httpx.Response(
                200,
                json=partial_json_response(
                    "id_1",
                    '{"translations":[{"french":"fr2","english":"en2"},{"french":"fr3","english":"en3"}, {"french":"fr4","english":"en4"}]}',
                ),
            ),
            httpx.Response(
                200,
                json=partial_json_response(
                    "id_2",
                    '{"translations":[{"french":"frA","english":"enA"},{"french":"frB","english":"enB"}]}',
                ),
            ),
        ]
    )

    translations = cli.generate_translations(record, client)

    assert translations == [("fr2", "en2"), ("fr3", "en3")]

    # 3 retries with the 3rd OK
    # First request returns 1 translation -> should be 2 so we retry
    # Second request returns no translation -> should be 2 so we retry
    # Third request returns 2 translations -> this is ok
    respx_mock.post("/responses").mock(
        side_effect=[
            httpx.Response(
                200,
                json=partial_json_response(
                    "id_1",
                    '{"translations":[{"french":"fr2","english":"en2"}]}',
                ),
            ),
            httpx.Response(
                200,
                json={
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "output": [
                        {
                            "id": "rs_0c8b0343bd64d781006971f5c6041c8194b28661972de6acc2",
                            "summary": [],
                            "type": "reasoning",
                        }
                    ],
                },
            ),
            httpx.Response(
                200,
                json=partial_json_response(
                    "id_2",
                    '{"translations":[{"french":"fr2","english":"en2"},{"french":"fr3","english":"en3"}]}',
                ),
            ),
        ]
    )

    translations = cli.generate_translations(record, client)

    assert translations == [("fr2", "en2"), ("fr3", "en3")]
    assert "No translations were returned by the model at attempt 2." in caplog.text
    assert (
        "Wrong number of translations returned by the model at attempt 1."
        in caplog.text
    )

    # Raise an error because we receive only one translation pair each
    # time we do a request to the API (at the 3rd attempt we raise an error)
    respx_mock.post("/responses").mock(
        return_value=httpx.Response(
            200,
            json=partial_json_response(
                "id_1",
                '{"translations":[{"french":"fr2","english":"en2"}]}',
            ),
        ),
    )

    with pytest.raises(ValueError, match="Wrong number of translations: 1."):
        translations = cli.generate_translations(record, client)

    # Raise an error because we receive no translation pair each time we
    # do a request to the API (at the 3rd attempt we raise an error)
    # This can happens if you use for instance gpt-5-nano with
    # a limited amount output token that entirely consumed by the
    # reasoning.
    # We're not using reasoning of gpt-5.2 but just in case.
    respx_mock.post("/responses").mock(
        return_value=httpx.Response(
            200,
            json={
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [
                    {
                        "id": "rs_0c8b0343bd64d781006971f5c6041c8194b28661972de6acc2",
                        "summary": [],
                        "type": "reasoning",
                    }
                ],
            },
        )
    )

    with pytest.raises(ValueError, match="No translations were returned by the model."):
        translations = cli.generate_translations(record, client)


@pytest.mark.respx(base_url="https://api.openai.com/v1/")
def test_generate_translations_request_logged_when_api_error_raised(
    respx_mock: MockRouter, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.INFO, logger="phrasebook_fr_to_en.cli")
    respx_mock.post("/responses").mock(
        return_value=httpx.Response(
            401, json={"error": {"message": "Incorrect API key provided"}}
        )
    )

    record = ("2025-12-15", "fr1", "en1")
    client = OpenAI(api_key="foo-api-key", max_retries=0)
    with pytest.raises(APIError):
        translations = cli.generate_translations(record, client)

    # Log httpx request
    assert "<Request('POST', 'https://api.openai.com/v1/responses')>" in caplog.text
    # Log httpx headers: Headers({'host': 'api.openai.com', ...})
    assert "Request headers -" in caplog.text
    assert "'host': 'api.openai.com'" in caplog.text
    # Ensure API key not log in headers
    assert "foo-api-key" not in caplog.text
    # Log httpx body request
    assert re.search(r"Request body - .*\"input\"\s*:\s*\"fr1 -> en1\"", caplog.text)


@pytest.mark.skipif(
    os.getenv("OPENAI_LIVE") != "1",
    reason="Requires OPENAI_LIVE=1. In that case, we do real call to OpenAI API.",
)
def test_generate_translations_real(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger="phrasebook_fr_to_en.cli")

    record = ("2025-12-15", "Il est beau.", "He is handsome.")
    client = OpenAI()
    translations = cli.generate_translations(record, client)

    # Raise an error if `translations` not valid against `TranslationList`
    TranslationList = conlist(tuple[str, str], min_length=2, max_length=2)
    TypeAdapter(TranslationList).validate_python(translations)

    assert (
        "Translations generated for record ('2025-12-15', 'Il est beau.', 'He is handsome.')"
        in caplog.text
    )
    assert (
        "using model gpt-5.2 and input 'Il est beau. -> He is handsome.'" in caplog.text
    )


@pytest.mark.skipif(
    os.getenv("OPENAI_LIVE") != "1",
    reason="Requires OPENAI_LIVE=1. In that case, we do real call to OpenAI API.",
)
def test_generate_audio_real(caplog: pytest.LogCaptureFixture, tmp_path: Path):
    caplog.set_level(logging.INFO, logger="phrasebook_fr_to_en.cli")

    record = cli.build_record(
        record_id=1,
        french="Il est beau.",
        english="He is handsome.",
        generated_from=pd.NA,
        date="2025-12-15",
    )
    client = OpenAI()
    media_dir = tmp_path
    audio_path = media_dir / record["audio_filename"]

    cli.generate_audio(record, media_dir, client)

    assert "Generating audio 'He is handsome.' for record 1" in caplog.text
    assert media_dir.exists()
    assert is_mp3(audio_path)
    assert f"Audio has been generated: {audio_path}." in caplog.text


@pytest.mark.skipif(
    os.getenv("OPENAI_LIVE") != "1",
    reason="Requires OPENAI_LIVE=1. In that case, we do real call to OpenAI API.",
)
def test_generate_img_real(caplog: pytest.LogCaptureFixture, tmp_path: Path):
    caplog.set_level(logging.INFO, logger="phrasebook_fr_to_en.cli")

    record = cli.build_record(
        record_id=1,
        french="Il est beau.",
        english="He is handsome.",
        generated_from=pd.NA,
        date="2025-12-15",
    )
    client = OpenAI()
    media_dir = tmp_path
    img_path = media_dir / record["img_filename"]

    cli.generate_img(record, media_dir, client)

    assert "Generating image 'He is handsome.' for record 1" in caplog.text
    assert media_dir.exists()
    assert is_png(img_path)
    assert f"Image has been generated: {img_path}." in caplog.text


@pytest.mark.skipif(
    os.getenv("OPENAI_LIVE") != "1",
    reason="Requires OPENAI_LIVE=1. In that case, we do real call to OpenAI API.",
)
def test_enrich_record_ok_real(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger="phrasebook_fr_to_en.cli")

    record = ("2025-12-15", "Il est beau.", "He is handsome.")
    media_dir = tmp_path
    client = OpenAI()

    records = cli.enrich_record(record, 10, media_dir, client)

    assert len(records) == 3
    for rec in records:
        audio_path = tmp_path / rec["audio_filename"]
        img_path = tmp_path / rec["img_filename"]
        assert is_mp3(audio_path)
        assert is_png(img_path)


@pytest.mark.skipif(
    os.getenv("OPENAI_LIVE") != "1",
    reason="Requires OPENAI_LIVE=1. In that case, we do real call to OpenAI API.",
)
def test_app_real(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger="phrasebook_fr_to_en.cli")

    phrasebook_path = tmp_path / "phrasebook.tsv"
    phrasebook_path.write_text("date\tfrench\tenglish\n2025-12-15\tbeau\thandsome")
    enriched_path = cli.enriched_path_func(phrasebook_path)

    result = runner.invoke(cli.app, [str(phrasebook_path)], catch_exceptions=False)

    assert result.exit_code == 0, result.stdout

    enriched_df = pd.read_csv(enriched_path, sep="\t", dtype="string")

    pd.testing.assert_index_equal(enriched_df.columns, pd.Index(cli.ENRICHED_COLUMNS))
    assert len(enriched_df) == 3

    media_dir = cli.media_dir_func(phrasebook_path)
    for audio_path in enriched_df["audio_filename"]:
        assert is_mp3(media_dir / audio_path)
    for img_filename in enriched_df["img_filename"]:
        assert is_png(media_dir / img_filename)
    assert "Record has been enriched: ('2025-12-15', 'beau', 'handsome')" in caplog.text
