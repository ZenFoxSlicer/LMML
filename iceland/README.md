# What The Data? — Iceland

Clean geothermal station exports with Python 3.10–3.12. No third-party dependencies are required.

The cleaner skips the variable comment/blank-line preamble, parses semicolon-delimited readings and decimal commas, repairs station-name encoding, converts Fahrenheit to Celsius, deduplicates by reading ID, and sorts numerically. Output temperatures have exactly one decimal place.

## Run

From this folder, place the evaluation export here as `test-input.csv`, then run:

```sh
python3 main.py
```

This writes `test-output.csv` in the current working directory.

## Practice check

```sh
cp train-input.csv test-input.csv
python3 main.py
python3 -c "from pathlib import Path; assert Path('test-output.csv').read_text(encoding='utf-8') == Path('train-output.csv').read_text(encoding='utf-8'); print('PASS')"
```

The comparison normalizes line endings, as the evaluator does. Only `main.py` is submitted to the platform. The flag is released by the platform after the held-back export passes.
