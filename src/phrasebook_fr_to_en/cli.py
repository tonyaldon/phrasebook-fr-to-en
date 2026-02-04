from __future__ import annotations
import os
import base64
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Annotated, Iterator
import contextlib
import typer
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer


__version__ = "0.3.0"

if TYPE_CHECKING:
    from pydantic import BaseModel
    import pandas as pd
    from openai import OpenAI

logger = logging.getLogger(__name__)

MEDIA_PREFIX = "phrasebook-fr-to-en-"

# See https://docs.ankiweb.net/importing/text-files.html#importing-media
ANKI_AUDIO_TEMPLATE = "[sound:{}]"
ANKI_IMG_TEMPLATE = '<img src="{}">'

ENRICHED_COLUMNS: list[str] = [
    "french",
    "english",
    "anki_audio",
    "anki_img",
    "generated_from",
    "id",
    "audio_filename",
    "img_filename",
    "date",
]

PHRASEBOOK_COLUMNS: list[str] = ["date", "french", "english"]


app = typer.Typer(pretty_exceptions_enable=False)


def enriched_path_func(phrasebook_path: Path) -> Path:
    return (phrasebook_path.parent / "enriched_phrasebook.tsv").absolute()


def phrasebook_dir_func(phrasebook_path: Path) -> Path:
    return phrasebook_path.parent.absolute()


def media_dir_func(phrasebook_path: Path) -> Path:
    return phrasebook_path.parent.absolute() / "media"


def read_phrasebook(phrasebook_path: Path) -> pd.DataFrame:
    import pandas as pd

    try:
        df = pd.read_csv(phrasebook_path, sep="\t", dtype="string")
    except FileNotFoundError as err:
        raise err
    except Exception as err:
        raise ValueError(f"Invalid file {phrasebook_path}: {err}")

    if list(df.columns) != PHRASEBOOK_COLUMNS:
        raise ValueError(
            f"Invalid header in {phrasebook_path}. Expected {PHRASEBOOK_COLUMNS}, got {list(df.columns)}"
        )

    return df


def read_enriched(enriched_path: Path) -> pd.DataFrame:
    import pandas as pd

    if not enriched_path.exists():
        return pd.DataFrame(columns=pd.Index(ENRICHED_COLUMNS), dtype="string")

    try:
        df = pd.read_csv(enriched_path, sep="\t", dtype="string")
    except Exception as err:
        raise ValueError(f"Invalid file {enriched_path}: {err}")

    if list(df.columns) != ENRICHED_COLUMNS:
        raise ValueError(
            f"Invalid header in {enriched_path}. Expected {ENRICHED_COLUMNS}, got {list(df.columns)}"
        )

    df["id"] = df["id"].astype("Int64")
    df["generated_from"] = df["generated_from"].astype("Int64")

    return df


@contextlib.contextmanager
def log_request_info_when_api_error_raised() -> Iterator[None]:
    from openai import APIError

    try:
        yield
    except APIError as exc:
        logger.error(f"{exc.request!r}")
        # It's safe to log httpx headers because its repr
        # sets 'authorization' to '[secure]' hidding our API key
        logger.error(f"Request headers - {exc.request.headers!r}")
        logger.error(f"Request body - {exc.request.content.decode()}")
        raise exc


def generate_translations(
    record_original: tuple[str, str, str], client: OpenAI
) -> list[tuple[str, str]]:
    from pydantic import BaseModel

    class Translation(BaseModel):
        french: str
        english: str

    class Translations(BaseModel):
        # DON'T USE: conlist(tuple[str, str], min_length=2, max_length=2)
        # This broke OpenAI API which generated outputs with 128,000 tokens.
        # Mostly, whitespaces and newlines.
        translations: list[Translation]

    _, french, english = record_original
    model = "gpt-5.2"
    instructions = """# Role and Objective
You are a bilingual (French/English) teacher specializing in practical language learning. Your task is to help expand a French-to-English phrasebook by creating relevant sentence pairs and highlighting key language aspects.

# Instructions
- For each prompt, you will get a French sentence and its English translation.
- Your tasks:
  1. Generate exactly two related English sentences, each with its French translation.
  2. Use these to show:
     - An English grammar point, or
     - Useful nouns, verbs, or
     - Alternative phrasing (formality, slang, etc.).
  3. Ensure all English examples are natural and suitable for daily use.

# Context
- The learner is a native French speaker advancing in English.
- The goal is to create a learner-friendly, practical phrasebook."""
    input_msg = f"{french} -> {english}"

    logger.info(f"Generating translations for record {record_original}")

    attempt = 1
    translations = []

    while not translations:
        with log_request_info_when_api_error_raised():
            response = client.responses.parse(
                model=model,
                instructions=instructions,
                input=input_msg,
                text_format=Translations,
                max_output_tokens=256,
            )

        # If we decided to use gpt-5-nano with the same low max_output_tokens,
        # tokens would be consumed by the reasoning, we would get
        # no text output, and this would result in output_parsed being None.
        if not response.output_parsed:
            if attempt < 3:
                logger.info(
                    f"No translations were returned by the model at attempt {attempt}."
                )
                attempt += 1
                continue
            else:
                raise ValueError(
                    f"No translations were returned by the model.\nResponse: {response.to_json()}"
                )

        translations = response.output_parsed.translations

        if (tlen := len(translations)) < 2:
            if attempt < 3:
                logger.info(
                    f"Wrong number of translations returned by the model at attempt {attempt}."
                )
                attempt += 1
                translations = []
                continue
            else:
                raise ValueError(
                    (
                        f"Wrong number of translations: {tlen}.  2 were expected.\n"
                        f"Response: {response.to_json()}"
                    )
                )

    logger.info(
        f"Translations generated for record {record_original} using model {model} and input '{input_msg}'"
    )

    return [(t.french, t.english) for t in translations][:2]


def generate_audio(record: dict[str, Any], media_dir: Path, client: OpenAI) -> None:
    id_record = record["id"]
    input_msg = record["english"]
    audio_path = media_dir / record["audio_filename"]

    media_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating audio '{input_msg}' for record {id_record}")
    with log_request_info_when_api_error_raised():
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="cedar",
            input=input_msg,
            instructions="Speak in a neutral General American accent at a natural conversational pace.  Use clear, natural intonation with a neutral tone, and avoid emotional coloring, character impressions, and whispering.",
        ) as response:
            response.stream_to_file(audio_path)

    logger.info(f"Audio has been generated: {audio_path}.")

    return None


def generate_img(record: dict[str, Any], media_dir: Path, client: OpenAI) -> None:
    id_record = record["id"]
    english = record["english"]
    img_path = media_dir / record["img_filename"]

    media_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating image '{english}' for record {id_record}")

    prompt = (
        f'Create a clean, minimal flat vector illustration representing: "{english}".\n'
        "Use 2-4 simple objects/characters maximum, solid colors, white background.\n"
        "No text, no letters, no numbers, no icons that resemble writing.\n"
    )

    with log_request_info_when_api_error_raised():
        response = client.images.generate(
            model="gpt-image-1.5",
            prompt=prompt,
            size="1024x1024",
            quality="low",
            output_format="png",
        )

    image_base64 = response.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)
    with open(img_path, "wb") as f:
        f.write(image_bytes)

    logger.info(f"Image has been generated: {img_path}.")

    return None


def next_id(enriched_df: pd.DataFrame) -> int:
    if enriched_df.empty:
        return 1

    max_id = int(enriched_df["id"].max())
    return max_id + 1


def build_record(
    record_id: int,
    generated_from: Any,
    date: str,
    french: str = "",
    english: str = "",
) -> dict[str, Any]:
    audio_filename = f"{MEDIA_PREFIX}{record_id}.mp3"
    img_filename = f"{MEDIA_PREFIX}{record_id}.png"
    return {
        "id": record_id,
        "french": french,
        "english": english,
        "anki_audio": ANKI_AUDIO_TEMPLATE.format(audio_filename),
        "anki_img": ANKI_IMG_TEMPLATE.format(img_filename),
        "generated_from": generated_from,
        "audio_filename": audio_filename,
        "img_filename": img_filename,
        "date": date,
    }


def enrich_record(
    record_original: tuple[str, str, str],
    next_id: int,
    media_dir: Path,
    client: OpenAI,
) -> list[dict[str, Any]]:
    import pandas as pd

    date, french, english = record_original

    try:
        translations = generate_translations(record_original, client)
    except Exception:
        logger.exception(
            f"Failed to generate translations while processing record {record_original}"
        )
        return []

    id_record_original = next_id
    new_records: list[dict[str, Any]] = [
        build_record(
            record_id=id_record_original,
            french=french,
            english=english,
            generated_from=pd.NA,
            date=date,
        )
    ]

    for i in range(len(translations)):
        new_records.append(
            build_record(
                record_id=id_record_original + i + 1,
                french=translations[i][0],
                english=translations[i][1],
                generated_from=id_record_original,
                date=date,
            )
        )

    try:
        for record in new_records:
            generate_audio(record, media_dir, client)
    except Exception:
        logger.exception(
            f"Failed to generate audios while processing record {record_original}"
        )
        return []

    try:
        for record in new_records:
            generate_img(record, media_dir, client)
    except Exception:
        logger.exception(
            f"Failed to generate images while processing record {record_original}"
        )
        return []

    return new_records


def save_new_records(
    new_records: list[dict[str, Any]], enriched_df: pd.DataFrame, enriched_path: Path
) -> pd.DataFrame:
    import pandas as pd

    new_df = pd.DataFrame(
        new_records, columns=pd.Index(ENRICHED_COLUMNS), dtype="string"
    )
    new_df["id"] = new_df["id"].astype("Int64")
    new_df["generated_from"] = new_df["generated_from"].astype("Int64")

    updated = (
        pd.concat([enriched_df, new_df], ignore_index=True)
        if not enriched_df.empty
        else new_df
    )
    updated.to_csv(enriched_path, sep="\t", index=False)
    return updated


def enrich_phrasebook(phrasebook_path: Path, client: OpenAI) -> bool:
    media_dir = media_dir_func(phrasebook_path)
    enriched_path = enriched_path_func(phrasebook_path)
    try:
        phrasebook_df = read_phrasebook(phrasebook_path)
        enriched_df = read_enriched(enriched_path)
    except Exception as err:
        logger.exception(err)
        return False

    existing_english: set[str] = (
        set(enriched_df["english"].dropna().to_list())
        if not enriched_df.empty
        else set()
    )

    for record_original in phrasebook_df.itertuples(index=False, name=None):
        _, _, english = record_original

        if english in existing_english:
            logger.info(f"Skip existing record: {record_original}")
            continue

        new_records = enrich_record(
            record_original, next_id(enriched_df), media_dir, client
        )
        if not new_records:
            return False

        try:
            enriched_df = save_new_records(new_records, enriched_df, enriched_path)
        except Exception:
            logger.exception(
                f"Failed to save enriched records from record {record_original} in file {enriched_path}"
            )
            return False

        existing_english.add(english)
        logger.info(f"Record has been enriched: {record_original} -> {enriched_path}")

    return True


def watch_phrasebook(phrasebook_path: Path, client: OpenAI) -> None:
    class Handler(FileSystemEventHandler):
        def on_modified(self, event: FileSystemEvent) -> None:
            if event.src_path == str(phrasebook_path):
                enrich_phrasebook(phrasebook_path, client)

    observer = Observer()
    observer.schedule(
        Handler(), str(phrasebook_dir_func(phrasebook_path)), recursive=False
    )
    observer.start()
    logger.info(f"Start watching file {phrasebook_path}")

    try:
        while True:
            time.sleep(1)
    finally:  # pragma: no cover
        observer.stop()
        observer.join()


def version_callback(version: bool):
    if version:
        print(f"phrasebook-fr-to-en {__version__}")
        raise typer.Exit()


def setup_logging(log_file: Path | None = None):
    log_format = "%(asctime)s %(levelname)s %(name)s %(message)s"
    log_datefmt = "%Y-%m-%d %H:%M:%S"
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            format=log_format, datefmt=log_datefmt, filename=log_file.absolute()
        )
    else:
        logging.basicConfig(format=log_format, datefmt=log_datefmt)

    logger.setLevel(logging.INFO)


@app.command()
def run(
    file: Annotated[
        Path,
        typer.Argument(
            help=(
                "Filename of the phrasebook to be enriched.  It must be a TSV format file (TAB separation) with the columns: date, french, english.  For instance:\n\n\n\n"
                "date        french          english\n\n"
                "2025-12-15  J'aime l'eau.   I like water.\n\n"
                "2025-12-16  Il fait froid.  It is cold.\n\n"
            )
        ),
    ],
    watch: Annotated[
        bool,
        typer.Option(
            "--watch",
            help="Watch the original phrasebook file for changes, and enrich any new record added to it.",
        ),
    ] = False,
    log_file: Annotated[
        Path | None,
        typer.Option(
            help="Log to this file if provided.  Default is stderr.",
        ),
    ] = None,
    version: Annotated[
        bool,
        typer.Option("--version", callback=version_callback, is_eager=True),
    ] = False,
) -> None:
    # We escape \[sound:...] because this is a reserved syntax for Rich.
    # If we don't, [sound:...] is removed from the help message.
    # But when we do this we get a SyntaxWarning, so we use raw string
    # r"""...""".
    r"""
    Enrich French to English phrasebooks with OpenAI API.

    -----

    [IMPORTANT] This program uses the OpenAI API with the following models:

    - https://platform.openai.com/docs/models/gpt-5.2
    - https://platform.openai.com/docs/models/gpt-4o-mini-tts
    - https://platform.openai.com/docs/models/gpt-image-1.5

    To use it, you need to register with OpenAI, be verified as an organization (required for the image model), and create an API key: https://platform.openai.com.

    Once you've done this, set OPENAI_API_KEY as an environment variable before you run the program, like this:

        $ export OPENAI_API_KEY=<your-api-key>

    -----

    This program takes a TSV (TAB separation) file as input.  Each row is a French to English translation.  It uses the following columns: date, french, english.

    For each translation (each row), two new related translations are generated.  The goal is to show:

    - An English grammar point, or
    - Useful nouns, verbs, or
    - Alternative phrasing (formality, slang, etc.).

    These new translations, along with the original, are saved in the file "enriched_phrasebook.tsv".  It sits next to your phrasebook file.  Records in your original phrasebook whose english field matches a record in the enriched phrasebook are skipped.

    Your original phrasebook is left unchanged.

    For all translations (original and AI generated), an English audio and an image are generated.  They are saved in a "media" directory next to the original phrasebook file.

    For instance, if "my-phrasebook.tsv" contains the following record
    (columns separated by tabs)

        date        french                english
        2025-12-15  Montez les escaliers. Climb the stairs.

    and you run the following commands:

        $ export OPENAI_API_KEY=<your-api-key>
        $ phrasebook-fr-to-en my-phrasebook.tsv

    This will produce the file "enriched_phrasebook.tsv" with AI generated translations.  It has the following columns: french, english, anki_audio, anki_img, generated_from, id, audio_filename, img_filename, date.

        french english anki_audio anki_img generated_from id audio_filename img_filename date
        Montez les escaliers. Climb the stairs. \[sound:phrasebook-fr-to-en-1.mp3] "<img src=""phrasebook-fr-to-en-1.png"">"  1 phrasebook-fr-to-en-1.mp3 phrasebook-fr-to-en-1.png 2025-11-15
        Prenez les escaliers, s'il vous plaît. Please take the stairs. \[sound:phrasebook-fr-to-en-2.mp3] "<img src=""phrasebook-fr-to-en-2.png"">" 1 2 phrasebook-fr-to-en-2.mp3 phrasebook-fr-to-en-2.png 2025-11-15
        Montez deux étages et tournez à gauche. Go up two floors and turn left. \[sound:phrasebook-fr-to-en-3.mp3] "<img src=""phrasebook-fr-to-en-3.png"">" 1 3 phrasebook-fr-to-en-3.mp3 phrasebook-fr-to-en-3.png 2025-11-15

    This also generates 3 audios and 3 images.

    Your directory then looks like this:

        .
        ├── enriched_phrasebook.tsv
        ├── my-phrasebook.tsv
        └── media
            ├── phrasebook-fr-to-en-1.mp3
            ├── phrasebook-fr-to-en-1.png
            ├── phrasebook-fr-to-en-2.mp3
            ├── phrasebook-fr-to-en-2.png
            ├── phrasebook-fr-to-en-3.mp3
            └── phrasebook-fr-to-en-3.png

    For Anki users.  Did you notice the columns "anki_audio" and "anki_img"?  They contain formatted fields for audio and image that you can use directly in your Anki decks:

        \[sound:phrasebook-fr-to-en-1.mp3]
        <img src="phrasebook-fr-to-en-1.png">

    This way you can import "enhanced_phrasebook.tsv" directly into Anki.  No changes are needed to get audio played and images displayed.

    Note that this only works:

    1) If you enable "Allow HTML" option when importing the enriched file,
    2) If you copy the audios and images from the "media" directory to your Anki "collection.media" directory.

    See Anki docs:

    - https://docs.ankiweb.net/importing/text-files.html#importing-media
    - https://docs.ankiweb.net/files.html
    """
    from openai import OpenAI, APIError

    setup_logging(log_file)

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("Set OPENAI_API_KEY environment variable to run the app.")
        raise typer.Exit(code=1)

    client = OpenAI()
    phrasebook_path = file.absolute()

    if not enrich_phrasebook(phrasebook_path, client):
        raise typer.Exit(code=1)

    if watch:  # pragma: no cover
        watch_phrasebook(phrasebook_path, client)
        return


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
