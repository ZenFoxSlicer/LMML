"""Remove personal details while keeping the rest of each message readable.

Upload this file on its own. It needs only Python's standard library.
"""

import json
import re
import unicodedata
from datetime import date
from pathlib import Path


def fold(text):
    """Make accents and capital letters irrelevant when looking for a match."""
    return ''.join(c for c in unicodedata.normalize('NFKD', text.lower())
                   if not unicodedata.combining(c))


def normalized_copy(text):
    # Keep a map back to the original message. We search the simple copy but
    # edit the original, so its spelling, accents, punctuation and lines survive.
    letters, starts, ends = [], [], []
    for i, character in enumerate(text):
        simple = fold(character)
        for c in simple:
            letters.append(c)
            starts.append(i)
            ends.append(i + 1)
        if not simple and ends:
            ends[-1] = i + 1
    return ''.join(letters), starts, ends


MONTHS = (
    ('january', 'jan', 'enero'), ('february', 'feb', 'febrero'),
    ('march', 'mar', 'marzo'), ('april', 'apr', 'abril'),
    ('may', 'mayo'), ('june', 'jun', 'junio'), ('july', 'jul', 'julio'),
    ('august', 'aug', 'agosto'), ('september', 'sep', 'sept', 'septiembre', 'setiembre'),
    ('october', 'oct', 'octubre'), ('november', 'nov', 'noviembre'),
    ('december', 'dec', 'diciembre'),
)
MONTH_NUMBER = {word: number for number, words in enumerate(MONTHS, 1) for word in words}
MONTH_PATTERN = '(?:' + '|'.join(sorted(MONTH_NUMBER, key=len, reverse=True)) + r')\.?'
DATE_PATTERNS = (
    r'\b(?P<year>\d{4})[-/.](?P<month>\d{1,2})[-/.](?P<day>\d{1,2})\b',
    r'\b(?P<day>\d{1,2})[-/.](?P<month>\d{1,2})[-/.](?P<year>\d{4})\b',
    rf'\b(?P<month>{MONTH_PATTERN})\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?[,\s]+(?P<year>\d{{4}})\b',
    rf'\b(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:de\s+)?(?P<month>{MONTH_PATTERN})\s+(?:de\s+)?(?P<year>\d{{4}})\b',
)

# These are clues about a person's role, not a list of training names.
NAME_START = (
    r'\b(?:my\s+(?:full\s+)?name\s+is|mi\s+nombre\s+es|me\s+llamo|'
    r'(?:passenger|customer|traveler|traveller|guest|contact|full)\s+name\s*[:=]|'
    r'(?:my|our|mi|su)\s+(?:son|daughter|wife|husband|partner|friend|mother|father|'
    r'sister|brother|spouse|child|colleague|hijo|hija|esposo|esposa|amigo|amiga)|'
    r'(?:travelling|traveling)\s+with|(?:passenger|guest|traveler|traveller)\s+named|'
    r'(?:name|nombre)\s*[:=])\s*'
)
NAME_PARTICLES = {'de', 'del', 'la', 'las', 'los', 'da', 'das', 'do', 'dos', 'van', 'von', 'y'}
NAME_STOP = set(('will would can could should is was has have had needs wants called says '
                 'and but for from with at to on in if please urgently urgent immediately '
                 'booking reservation reference phone email account number cedula date '
                 'born lives living travelling traveling the a an el una un es soy '
                 'viaja viajara necesita thanks regards sincerely hello hi dear').split())
NAME_STOP.update(MONTH_NUMBER)


class MessageRedactor:
    def __init__(self, text):
        self.original = text
        self.text, self.starts, self.ends = normalized_copy(text)
        self.spans = []
        self.aliases = {}
        # The practice data uses letters as well as digits in PV references.
        # Protect the actual characters; never replace or rewrite a reference.
        self.bookings = [m.span() for m in re.finditer(r'\bpv-[a-z0-9]{5}\b', self.text)]

    def remember(self, start, end, category):
        """Mark a personal detail and remember it so repeats can be removed."""
        if start >= end or any(start < b and end > a for a, b in self.bookings):
            return
        self.spans.append((start, end, category))
        self.alias(self.text[start:end], category)

    def alias(self, value, category):
        # Remember values without separators. Later we can match both
        # "8712-3456" and "87123456", or both "José" and "JOSE".
        compact = ''.join(c for c in fold(value) if c.isalnum())
        if len(compact) >= 2:
            self.aliases.setdefault(compact, category)

    def find_numbers_and_emails(self):
        # Recognizable formats can be found even without a label beside them.
        patterns = (
            ('EMAIL', r"[a-z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?\.[a-z]{2,}"),
            ('ACCOUNT', r'(?<!\w)cr(?:[\s.-]*\d){20}(?!\d)'),
            ('ID', r'(?<!\w)[1-9][ .-]?\d{4}[ .-]?\d{4}(?!\w)'),
            ('PHONE', r'(?<!\w)(?:(?:\+|00)?506[\s().-]*)?[2-9](?:[\s().-]*\d){7}(?!\w)'),
        )
        for category, pattern in patterns:
            for m in re.finditer(pattern, self.text):
                # Eight digits can also be an ISO travel date. Don't mistake
                # 2027-07-21 for a phone number just because it has eight digits.
                if category == 'PHONE' and re.fullmatch(r'\d{4}[-/. ]\d{2}[-/. ]\d{2}', m.group()):
                    continue
                self.remember(*m.span(), category)
                if category == 'EMAIL':
                    # Removing the email is not enough: its username might
                    # appear elsewhere without the @domain part.
                    self.alias(m.group().split('@')[0], 'EMAIL')
                elif category == 'PHONE':
                    digits = re.sub(r'\D', '', m.group())
                    self.alias(digits[-8:], 'PHONE')
                    self.alias('506' + digits[-8:], 'PHONE')
                    self.alias('00506' + digits[-8:], 'PHONE')

    def name_at(self, position):
        """Read the person's name after a clue such as 'My name is'."""
        words = list(re.finditer(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*", self.text[position:]))
        accepted = []
        cursor = position
        for word in words[:8]:
            start, end = position + word.start(), position + word.end()
            gap = self.text[cursor:start]
            if gap.strip() or '\n' in gap or word.group() in NAME_STOP:
                break
            accepted.append((start, end, word.group()))
            cursor = end
        # Don't leave a dangling "de la" attached to a name.
        while accepted and accepted[-1][2] in NAME_PARTICLES:
            accepted.pop()
        if not accepted:
            return
        start, end = accepted[0][0], accepted[-1][1]
        self.remember(start, end, 'NAME')
        # First names and surnames must disappear elsewhere too.
        for _, _, word in accepted:
            if word not in NAME_PARTICLES:
                self.alias(word, 'NAME')
                for part in re.split("[-'’]", word):
                    if len(part) >= 3 and part not in NAME_PARTICLES:
                        self.alias(part, 'NAME')

    def find_names(self):
        for m in re.finditer(NAME_START, self.text):
            self.name_at(m.end())
        # Signatures can introduce a name even when there is no "My name is".
        signature = r'\b(?:kind regards|best regards|regards|sincerely|yours truly|atentamente|saludos|signed)(?:[ \t]*[,!:][ \t\n]*|[ \t]*\n[ \t]*)'
        for m in re.finditer(signature, self.text):
            self.name_at(m.end())
        # "I am Ana López" is a name; "I am requesting a refund" is not.
        # Check the original capitalization before accepting this weaker clue.
        for m in re.finditer(r"\b(?:i am|i'm|soy|this is)\s+", self.text):
            tail = self.original[self.starts[m.end()]:] if m.end() < len(self.starts) else ''
            if re.match(r"[A-ZÀ-ÖØ-Þ][^\W\d_]+\s+[A-ZÀ-ÖØ-Þ][^\W\d_]+", tail):
                self.name_at(m.end())

    def find_addresses(self):
        # Stop at the end of the address, not at the end of the whole message.
        # In particular, keep "to our booking", urgency words and PV references.
        stop = (r'(?=\s+(?:to (?:our|my|the) booking|and (?:keep|please|call|send|'
                r'confirm|the|i|my)|keep reference|reference|please|asap|urgently)\b|'
                r'\s+pv-[a-z0-9]{5}\b|[;!\n]|\.(?:\s|$)|$)')
        street = r'\b(?:calle|avenida|av\.|barrio|urbanizacion|residencial|carretera|apartado postal)\s+(?=[a-z0-9])'
        context = (r'\b(?:my (?:postal |mailing |home )?address is|'
                   r'(?:postal |mailing |home )?address\s*[:=]|'
                   r'mi direccion es|direccion\s*[:=]|'
                   r'(?:i live|we live|i reside) (?:at|in)|'
                   r'pickup (?:address|location)\s*[:=])\s*')
        for pattern in (street + r'.+?' + stop, context + r'(?P<address>.+?)' + stop):
            for m in re.finditer(pattern, self.text):
                start, end = m.span('address') if 'address' in m.re.groupindex else m.span()
                while end > start and self.text[end - 1] in ' ,\t':
                    end -= 1
                value = self.text[start:end]
                self.remember(start, end, 'ADDRESS')
                # A repeated street or landmark is still personal data.
                for chunk in value.split(','):
                    chunk = chunk.strip()
                    if len(chunk) >= 4:
                        self.alias(chunk, 'ADDRESS')
                    # "Frente al Supermercado La Esquina" may return later as
                    # just "Supermercado La Esquina" or "La Esquina".
                    landmark = re.sub(
                        r'^(?:(?:\d+\s*(?:m|metros|meters)\s+)?'
                        r'(?:norte|sur|este|oeste|north|south|east|west)\s+)?'
                        r'(?:(?:contiguo|frente|junto|cerca)\s+)?'
                        r'(?:(?:a|de)\s+(?:la|el|los|las)\s+|(?:al|del)\s+)', '', chunk)
                    if landmark != chunk and len(landmark) >= 4:
                        self.alias(landmark, 'ADDRESS')
                    distinctive = re.sub(r'^(?:supermercado|panaderia|iglesia|parque)\s+', '', landmark)
                    if distinctive != landmark and len(distinctive) >= 5:
                        self.alias(distinctive, 'ADDRESS')

    def find_birthdays(self):
        for pattern in DATE_PATTERNS:
            for m in re.finditer(pattern, self.text):
                # A travel date is useful, so a date becomes private only when
                # it is introduced as a birthday (or repeats a known birthday).
                before = self.text[max(0, m.start() - 90):m.start()]
                if not re.search(r'\b(?:date of birth|birth\s*date|birthday|dob|born|fecha de nacimiento|naci|nacio)\b[^.!?\n]*$', before):
                    continue
                month_text = m['month'].rstrip('.')
                month = MONTH_NUMBER.get(month_text) if not month_text.isdigit() else int(month_text)
                day, year = int(m['day']), int(m['year'])
                if month > 12 and day <= 12:
                    day, month = month, day
                try:
                    date(year, month, day)
                except ValueError:
                    continue
                self.remember(*m.span(), 'DOB')
                # Remember the same birthday in the other accepted formats.
                for d in {str(day), f'{day:02d}'}:
                    for mo in {str(month), f'{month:02d}'}:
                        for variant in (f'{d}/{mo}/{year}', f'{mo}/{d}/{year}', f'{year}-{mo}-{d}'):
                            self.alias(variant, 'DOB')
                    for name in MONTHS[month - 1]:
                        for variant in (f'{name} {d} {year}', f'{d} {name} {year}', f'{d} de {name} de {year}'):
                            self.alias(variant, 'DOB')
                        for suffix in ('st', 'nd', 'rd', 'th'):
                            self.alias(f'{name} {d}{suffix} {year}', 'DOB')
                            self.alias(f'{d}{suffix} {name} {year}', 'DOB')

    def remove_repeats(self):
        # Search a separator-free copy. The position map tells us exactly
        # which original characters to remove, including any separators.
        compact, positions = [], []
        for i, c in enumerate(self.text):
            if c.isalnum():
                compact.append(c)
                positions.append(i)
        compact = ''.join(compact)
        for value, category in list(self.aliases.items()):
            offset = 0
            while True:
                found = compact.find(value, offset)
                if found < 0:
                    break
                start = positions[found]
                end = positions[found + len(value) - 1] + 1
                offset = found + 1
                # "Ana" should not erase the middle of "banana".
                if start and self.text[start - 1].isalnum():
                    continue
                if end < len(self.text) and self.text[end].isalnum():
                    continue
                if not any(start < b and end > a for a, b in self.bookings):
                    self.spans.append((start, end, category))

    def redact(self):
        self.find_numbers_and_emails()
        self.find_names()
        self.find_addresses()
        self.find_birthdays()
        self.remove_repeats()
        # A phone-like fragment can sit inside a longer account number. Merge
        # overlapping marks so one complete value gets one placeholder.
        merged = []
        for start, end, category in sorted(self.spans, key=lambda item: (item[0], -item[1])):
            if merged and start < merged[-1][1]:
                old_start, old_end, old_category = merged[-1]
                merged[-1] = (old_start, max(old_end, end), old_category)
            else:
                merged.append((start, end, category))
        pieces, cursor = [], 0
        for start, end, category in merged:
            a, b = self.starts[start], self.ends[end - 1]
            pieces.extend((self.original[cursor:a], '[' + category + ']'))
            cursor = b
        pieces.append(self.original[cursor:])
        return ''.join(pieces) or '[REDACTED]'


def main():
    # The evaluator supplies test-input.json in the working directory.
    # The fallback is only for local practice in a separate working copy.
    if Path('test-input.json').is_file():
        source, destination = 'test-input.json', 'test-output.json'
    else:
        source, destination = 'train-input.json', 'train-output.json'
        if Path(destination).exists():
            raise FileExistsError('Keep the golden train-output.json safe: run practice in a fresh directory.')
    data = json.loads(Path(source).read_text(encoding='utf-8-sig'))
    result = {document['id']: MessageRedactor(document['text']).redact()
              for document in data['documents']}
    Path(destination).write_text(json.dumps({'redacted_documents': result},
                                           ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
