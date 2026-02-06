# phrasebook-fr-to-en

`phrasebook-fr-to-en` is a CLI that uses the OpenAI API to enrich
French to English phrasebooks with AI generated translations, audios,
and images.

## Installing and running

It's a Python program.  You can install it as an `uv` tool like this:

```
uv tool install phrasebook-fr-to-en
```

After you generate an OpenAI API key, and with your original French to
English translations in the file `my-phrasebook.tsv`, you can generate
new ones, along with audios and images, by running this:

```
export OPENAI_API_KEY=<your-api-key>
phrasebook-fr-to-en my-phrasebook.tsv
```

This creates the file `enriched_phrasebook.tsv` with all translations.
It also saves the audios and images in the directory `media`.
Both `enriched_phrasebook.tsv` and `media` sit next to your phrasebook
file.

Your original phrasebook is left unchanged.

Run the following to list the options and their documentation:

```
phrasebook-fr-to-en --help
```

## OpenAI API [IMPORTANT]

This program uses the OpenAI API with the following models:

- https://platform.openai.com/docs/models/gpt-5.2
- https://platform.openai.com/docs/models/gpt-4o-mini-tts
- https://platform.openai.com/docs/models/gpt-image-1.5

To use it, you need to register with OpenAI.  You also need to be
verified as an organization (required for the image model).  Then
create an API key: https://platform.openai.com.

Once you've done this, set `OPENAI_API_KEY` as an environment variable
before you run the program, like this:

```
export OPENAI_API_KEY=<your-api-key>
```

## Format of your original phrasebook

Your original phrasebook must be a TSV file (TAB separation) with the
columns `date`, `french`, and `english`, like in this example:

```
date	french	english
2025-11-15	Montez les escaliers.	Climb the stairs.
2025-11-16	Il est beau.	He is handsome.
2025-01-17	Il est moche.	He is ugly.
```

## Full example

`phrasebook-fr-to-en` takes a TSV (TAB separation) file as input.
Each row is a French to English translation.  It uses the following
columns: `date`, `french`, `english`.

For each translation (each row), two new related translations are
generated.  The goal is to show:

- An English grammar point, or
- Useful nouns, verbs, or
- Alternative phrasing (formality, slang, etc.).

These new translations, along with the original, are saved in the file
`enriched_phrasebook.tsv`.  It sits next to your phrasebook file.
Records in your original phrasebook whose english field matches a
record in the enriched phrasebook are skipped.

Your original phrasebook is left unchanged.

For all translations (original and AI generated), an English audio and
an image are generated.  They are saved in a `media` directory next to
the original phrasebook file.

For instance, if `my-phrasebook.tsv` contains the following record
(columns separated by tabs)


|date|french|english|
|-----|-----|-----|
|2025-11-15|Montez les escaliers.|Climb the stairs.|


and you run the following commands:

```
export OPENAI_API_KEY=<your-api-key>
phrasebook-fr-to-en my-phrasebook.tsv
```

This produces the file `enriched_phrasebook.tsv` with AI generated
translations.  It has the following columns: `french`, `english`,
`anki_audio`, `anki_img`, `generated_from`, `id`, `audio_filename`,
`img_filename`, `date`.

|french|english|anki_audio|anki_img|generated_from|id|audio_filename|img_filename|date|
|-----|-----|-----|-----|-----|-----|-----|-----|-----|
|Montez les escaliers.|Climb the stairs.|[sound:phrasebook-fr-to-en-1.mp3]|"<img src=""phrasebook-fr-to-en-1.png"">"||1|phrasebook-fr-to-en-1.mp3|phrasebook-fr-to-en-1.png|2025-11-15|
|Montez les escaliers jusqu’au premier étage.|Climb the stairs up to the first floor.|[sound:phrasebook-fr-to-en-2.mp3]|"<img src=""phrasebook-fr-to-en-2.png"">"|1|2|phrasebook-fr-to-en-2.mp3|phrasebook-fr-to-en-2.png|2025-11-15|
|Prenez les escaliers, c’est juste à gauche.|Take the stairs; it’s just on the left.|[sound:phrasebook-fr-to-en-3.mp3]|"<img src=""phrasebook-fr-to-en-3.png"">"|1|3|phrasebook-fr-to-en-3.mp3|phrasebook-fr-to-en-3.png|2025-11-15|

This also generates 3 audios and 3 images.

Your directory then looks like this:

```
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
```

## For Anki users

In the previous example, did you notice the columns `anki_audio` and
`anki_img`?  They contain formatted fields for audio and image that
you can use directly in your [Anki](https://apps.ankiweb.net/) decks:

```
[sound:phrasebook-fr-to-en-1.mp3]
<img src="phrasebook-fr-to-en-1.png">
```

This way you can import `enhanced_phrasebook.tsv` directly into Anki.
No changes are needed to get audio played and images displayed.

Note that this only works:

1. If you enable "Allow HTML" option when importing the enriched file,
2. If you copy the audios and images from the `media` directory to
your Anki `collection.media` directory.

See Anki docs:

- https://docs.ankiweb.net/importing/text-files.html#importing-media
- https://docs.ankiweb.net/files.html

## Dev

### Installing and running from source

To run `phrasebook-fr-to-en` from the source, run this:

```
uv run src/phrasebook_fr_to_en/cli.py phrasebook.tsv
```

Alternatively, `phrasebook-fr-to-en` can be installed as an `uv` tool
from the source like this:

```
uv tool install .
```

Then run it like this:

```
phrasebook-fr-to-en my-phrasebook.tsv
```

### Running the tests

Run the tests like this:

```
uv run pytest
```

To run the tests with real calls to the OpenAI API, run this:

```
OPENAI_LIVE=1 uv run pytest
```

For this to work, the `OPENAI_API_KEY` environment variable must be
set using an OpenAI API key.  This variable can also be declared in a
`.env` file.

### Test coverage

This software has 100% test coverage.

To check this, you can run the following commands:

```
OPENAI_LIVE=1 uv run coverage run -m pytest
uv run coverage report
```

As mentioned above, `OPENAI_API_KEY` must be set prior to running the
tests with coverage.

### Linter + formating

`ty` and `ruff` must be installed first:

```
ty check
ruff format
```
