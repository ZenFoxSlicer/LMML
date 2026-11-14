"""Clean test-input.csv into test-output.csv using only the standard library."""

import csv


def restore_name(name):
    """Undo UTF-8 bytes misread as Latin-1 or Windows-1252, even in fragments."""
    while True:
        byte_values = []
        for character in name:
            if ord(character) <= 255:
                byte_values.append(ord(character))
            else:
                try:
                    byte_values.append(character.encode("cp1252")[0])
                except UnicodeEncodeError:
                    byte_values.append(None)

        repaired = []
        position = 0
        while position < len(name):
            first = byte_values[position]
            width = 0
            if first is not None:
                if 0xC2 <= first <= 0xDF:
                    width = 2
                elif 0xE0 <= first <= 0xEF:
                    width = 3
                elif 0xF0 <= first <= 0xF4:
                    width = 4
            chunk = byte_values[position : position + width]
            if width and len(chunk) == width and None not in chunk:
                try:
                    decoded = bytes(chunk).decode("utf-8")
                except UnicodeDecodeError:
                    pass
                else:
                    repaired.append(decoded)
                    position += width
                    continue
            repaired.append(name[position])
            position += 1

        result = "".join(repaired)
        if result == name:
            return result
        # Each successful repair shortens the name, so this loop terminates.
        name = result


def main():
    readings = {}
    with open("test-input.csv", encoding="utf-8-sig", newline="") as source:
        lines = (line for line in source if line.strip() and not line.lstrip().startswith("#"))
        for row in csv.DictReader(lines, delimiter=";"):
            reading_id = int(row["reading_id"])
            temperature = float(row["temperature_raw"].replace(",", "."))
            unit = row["unit"].strip().upper()
            if unit == "F":
                temperature = (temperature - 32) * 5 / 9
            elif unit != "C":
                raise ValueError("Unknown temperature unit: " + unit)

            # Retransmissions can differ in encoding, number format, or unit.
            readings.setdefault(reading_id, (
                reading_id,
                restore_name(row["station_name"]),
                f"{temperature:.1f}",
                row["reading_date"],
            ))

    with open("test-output.csv", "w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, quoting=csv.QUOTE_NONE, quotechar=None, lineterminator="\n")
        writer.writerow(("reading_id", "station_name", "temperature_c", "reading_date"))
        writer.writerows(readings[key] for key in sorted(readings))


if __name__ == "__main__":
    main()
