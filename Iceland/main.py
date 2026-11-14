# ICELAND: clean the damaged station table.
# 1. Read test-input.csv, skipping its opening comments and blank lines.
# 2. Repair broken station names and convert temperatures to Celsius.
# 3. Keep one row per reading ID, then sort by that numeric ID.
# 4. Write test-output.csv in exactly the required format.
# Start reading at main(); restore_name() handles the encoding repair.

"""Clean test-input.csv into test-output.csv using only the standard library."""

import csv


# Sometimes one real letter was mistaken for several characters.
# For example, the broken text MÃ½vatn should become Mývatn.
# Try turning those characters back into bytes, then reading the bytes correctly.
def restore_name(name):
    """Undo UTF-8 bytes misread as Latin-1 or Windows-1252, even in fragments."""
    # A name may have been damaged more than once, so repeat until nothing changes.
    while True:
        # A byte is a number from 0 to 255. Rebuild the possible original bytes.
        # None means a character cannot be converted by these old encodings.
        byte_values = []
        for character in name:
            # ord() gives a character its number. Latin-1 uses that number directly.
            if ord(character) <= 255:
                byte_values.append(ord(character))
            else:
                try:
                    # Windows-1252 assigns some other characters to single bytes.
                    byte_values.append(character.encode("cp1252")[0])
                except UnicodeEncodeError:
                    byte_values.append(None)

        # Walk through the name and collect repaired pieces in order.
        repaired = []
        position = 0
        while position < len(name):
            first = byte_values[position]
            width = 0
            # The first byte of a UTF-8 letter tells us whether it needs 2, 3, or 4 bytes.
            # The numbers starting with 0x are just byte values written in hexadecimal.
            if first is not None:
                if 0xC2 <= first <= 0xDF:
                    width = 2
                elif 0xE0 <= first <= 0xEF:
                    width = 3
                elif 0xF0 <= first <= 0xF4:
                    width = 4
            # Take that many possible bytes, but only try decoding a complete group.
            chunk = byte_values[position : position + width]
            if width and len(chunk) == width and None not in chunk:
                try:
                    # If these bytes form a valid UTF-8 letter, this recovers that letter.
                    decoded = bytes(chunk).decode("utf-8")
                except UnicodeDecodeError:
                    pass
                else:
                    repaired.append(decoded)
                    # Skip all the broken characters that were replaced by the recovered letter.
                    position += width
                    continue
            # If repair did not work here, keep the original character. Do not delete it.
            repaired.append(name[position])
            position += 1

        # Join the pieces. Stop if this pass could not repair anything else.
        result = "".join(repaired)
        if result == name:
            return result
        # Each successful repair shortens the name, so this loop terminates.
        name = result


def main():
    # A dictionary stores each reading under its ID, making duplicates easy to skip.
    readings = {}
    # utf-8-sig also accepts an optional invisible marker at the start of the file.
    # with automatically closes the file when we finish reading it.
    with open("test-input.csv", encoding="utf-8-sig", newline="") as source:
        # Ignore empty lines and lines whose first non-space character is #.
        lines = (line for line in source if line.strip() and not line.lstrip().startswith("#"))
        # Use semicolons to separate fields. Each row becomes a dictionary
        # whose keys come from the header, such as station_name and unit.
        for row in csv.DictReader(lines, delimiter=";"):
            # Read the ID as a number so 2 sorts before 10.
            # Replace a decimal comma with a dot before reading the temperature.
            reading_id = int(row["reading_id"])
            temperature = float(row["temperature_raw"].replace(",", "."))
            unit = row["unit"].strip().upper()
            # Fahrenheit to Celsius: subtract 32, multiply by 5, then divide by 9.
            if unit == "F":
                temperature = (temperature - 32) * 5 / 9
            elif unit != "C":
                raise ValueError("Unknown temperature unit: " + unit)

            # Retransmissions can differ in encoding, number format, or unit.
            # setdefault keeps the first row for an ID and ignores later copies.
            # The format .1f writes exactly one decimal place, including a trailing zero.
            readings.setdefault(reading_id, (
                reading_id,
                restore_name(row["station_name"]),
                f"{temperature:.1f}",
                row["reading_date"],
            ))

    # Write a new UTF-8 file. QUOTE_NONE prevents extra quotation marks.
    with open("test-output.csv", "w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, quoting=csv.QUOTE_NONE, quotechar=None, lineterminator="\n")
        writer.writerow(("reading_id", "station_name", "temperature_c", "reading_date"))
        # Sort the dictionary keys numerically and write the matching rows in order.
        writer.writerows(readings[key] for key in sorted(readings))


# Run main() when this file is executed directly, but not when it is imported.
if __name__ == "__main__":
    main()
