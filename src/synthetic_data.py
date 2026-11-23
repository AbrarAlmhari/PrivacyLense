"""Generator for the synthetic evaluation documents.

Why synthetic documents? The project never uses real personal documents.
Instead we generate fictional letters, emails, CVs, forms and chats from
templates. Because *we* insert every sensitive value, we know its exact
position, so the gold labels are created automatically and are always correct.

These documents are used ONLY for evaluation (never for training):
one third is a development set (for tuning rules and thresholds) and two
thirds are a held-out test set.

Everything is fictional: names are random combinations, emails use the
reserved ``example.com`` domains, phone numbers and IDs are random digits.

Run it with:  ``python scripts/03_make_synthetic_docs.py``
"""

from __future__ import annotations

import json
import logging
import random
import re
import textwrap
from pathlib import Path

import pymupdf

from src.config import resolve_path
from src.file_processor import clean_text

logger = logging.getLogger(__name__)

# ------------------------------------------------------------ fictional values
FIRST_NAMES = [
    "Sara", "Noura", "Fatimah", "Reem", "Lama", "Hessa", "Maha", "Dana", "Abdullah", "Mohammed",
    "Khalid", "Faisal", "Omar", "Yousef", "Abdulrahman", "Turki", "Emma", "Olivia", "James", "Daniel",
    "Sophia", "Lucas", "Priya", "Arjun", "Mei", "Kenji", "Amina", "Hassan", "Layla", "Ibrahim",
]
LAST_NAMES = [
    "Al-Qahtani", "Alharbi", "Al-Dossary", "Alghamdi", "Al-Otaibi", "Alzahrani", "Al-Shehri", "Almutairi",
    "Bakr", "Hakim", "Johnson", "Williams", "Anderson", "Martinez", "Nguyen", "Tanaka", "Sharma", "Patel",
    "Okafor", "Rossi", "Haddad", "Nasser", "Mansour", "Khoury", "Brown", "Wilson",
]
ORGANIZATIONS = [
    "Najd Horizon Trading", "Blue Falcon Logistics", "Al Waha Medical Center", "Desert Bloom Technologies",
    "Gulf Crescent Insurance", "Red Sea Analytics", "Palm Gate Consulting", "Eastern Dunes Academy",
    "Silverline Software", "Harbor Point Bank", "Greenfield Pharmacy", "Summit Ridge Engineering",
    "Bright Path Institute", "Oasis Travel Agency", "Crescent Valley Hospital", "Northgate Foods",
]
LOCATIONS = [
    "Riyadh", "Jeddah", "Dammam", "Al Khobar", "Dhahran", "Medina", "Abha", "Tabuk", "Saudi Arabia",
    "Dubai", "Cairo", "Amman", "Kuwait City", "London", "Toronto", "Chicago", "Singapore", "Berlin",
]
STREETS = [
    "King Fahd Road", "Prince Sultan Street", "Olaya Street", "King Abdullah Road", "Tahlia Street",
    "Maple Avenue", "Oak Street", "Cedar Lane", "Park Avenue", "Corniche Road",
]
EMAIL_DOMAINS = ["example.com", "example.org", "example.net", "mail.example.com"]
DATES = ["12 March 2026", "2026-04-18", "05/09/2026", "1 June 2026", "23 October 2026", "2026-01-07"]
AMOUNTS = ["1,250.00 SAR", "SAR 480", "$75.50", "3,900 SAR", "USD 120.00", "640.75 SAR"]
ORDER_NUMBERS = ["48213", "A-7731", "INV-2045", "REF-88", "90417", "PO-3312"]


def _digits(rng: random.Random, count: int) -> str:
    return "".join(rng.choice("0123456789") for _ in range(count))


def _make_person(rng: random.Random) -> str:
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def _make_email(rng: random.Random, person: str | None) -> str:
    person = person or _make_person(rng)
    first, last = person.lower().replace("-", "").split(" ", 1)
    style = rng.choice(["{f}.{l}", "{f}{l}", "{f}_{l}", "{f[0]}{l}", "{f}.{l}{n}"])
    local = style.format(f=first, l=last.replace(" ", ""), n=rng.randint(1, 99))
    return f"{local}@{rng.choice(EMAIL_DOMAINS)}"


def _make_phone(rng: random.Random) -> str:
    body = "5" + _digits(rng, 8)  # Saudi mobile numbers start with 5
    return rng.choice(
        [
            f"0{body}",
            f"0{body[:2]} {body[2:5]} {body[5:]}",
            f"+966{body}",
            f"+966 {body[:2]} {body[2:5]} {body[5:]}",
            f"00966{body}",
            f"+44 20 7946 0{_digits(rng, 3)}",
            f"(555) 01{_digits(rng, 1)}-{_digits(rng, 4)}",
            f"+1 555-01{_digits(rng, 1)}-{_digits(rng, 4)}",
        ]
    )


def _make_id(rng: random.Random) -> str:
    return rng.choice("12") + _digits(rng, 9)  # Saudi-style: 10 digits, starts with 1 or 2


def _make_card(rng: random.Random) -> str:
    """Random 16-digit number with a valid Luhn check digit (test prefix 4111)."""
    partial = "4111" + _digits(rng, 11)
    total = 0
    for position, char in enumerate(reversed(partial)):
        digit = int(char)
        if position % 2 == 0:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    number = partial + str((10 - total % 10) % 10)
    return rng.choice([number, " ".join(number[i : i + 4] for i in range(0, 16, 4))])


def _make_iban(rng: random.Random) -> str:
    number = "SA" + _digits(rng, 22)
    return rng.choice([number, " ".join(number[i : i + 4] for i in range(0, 24, 4))])


# Slot name -> (gold category or None for harmless "distractor" values, maker)
SLOT_MAKERS = {
    "PERSON": ("PERSON", lambda rng, ctx: _make_person(rng)),
    "ORG": ("ORG", lambda rng, ctx: rng.choice(ORGANIZATIONS)),
    "LOCATION": ("LOCATION", lambda rng, ctx: rng.choice(LOCATIONS)),
    "ADDRESS": ("ADDRESS", lambda rng, ctx: f"{rng.randint(10, 9999)} {rng.choice(STREETS)}"),
    "EMAIL": ("EMAIL", lambda rng, ctx: _make_email(rng, ctx)),
    "PHONE": ("PHONE", lambda rng, ctx: _make_phone(rng)),
    "ID": ("ID_LIKE", lambda rng, ctx: _make_id(rng)),
    "CARD": ("CREDIT_CARD", lambda rng, ctx: _make_card(rng)),
    "IBAN": ("IBAN", lambda rng, ctx: _make_iban(rng)),
    "DATE": (None, lambda rng, ctx: rng.choice(DATES)),
    "AMOUNT": (None, lambda rng, ctx: rng.choice(AMOUNTS)),
    "ORDER": (None, lambda rng, ctx: rng.choice(ORDER_NUMBERS)),
}

# ------------------------------------------------------------------ templates
# {PERSON_1} used twice gives the SAME name twice. {EMAIL_1} is built from
# {PERSON_1}. Dates, amounts and order numbers are distractors: they look
# "number-like" but must NOT be detected.
TEMPLATES: dict[str, str] = {
    "formal_letter": """{DATE_1}

Dear {PERSON_1},

We are pleased to confirm your appointment as Data Analyst at {ORG_1}. Your first working day is {DATE_2} at our {LOCATION_1} office, located at {ADDRESS_1}.

Please bring a copy of your national ID ({ID_1}) and send your bank details to {EMAIL_2} before the start date. If you have any questions, call our HR coordinator {PERSON_2} on {PHONE_2}.

Sincerely,
{PERSON_3}
Head of Human Resources, {ORG_1}""",
    "email": """From: {EMAIL_1}
To: {EMAIL_2}
Subject: Updated contact details

Hi {PERSON_2},

I have moved to {LOCATION_1} and my new mobile number is {PHONE_1}. Please update the records at {ORG_1} and remove my old number.

The invoice {ORDER_1} for {AMOUNT_1} was paid on {DATE_1}, so nothing else is pending from my side.

Thanks,
{PERSON_1}""",
    "cv": """CURRICULUM VITAE

Name: {PERSON_1}
Email: {EMAIL_1}
Mobile: {PHONE_1}
Address: {ADDRESS_1}, {LOCATION_1}

EXPERIENCE
Software Engineer at {ORG_1}, {LOCATION_2} (2023 - 2026). Built internal reporting tools and reduced processing time by 40 percent.
Intern at {ORG_2} (2022). Supported the data quality team.

EDUCATION
Bachelor of Science in Artificial Intelligence, graduated {DATE_1}.

REFERENCES
{PERSON_2}, Engineering Manager, {ORG_1}, {EMAIL_2}""",
    "registration_form": """STUDENT REGISTRATION FORM

Full name: {PERSON_1}
National ID / Iqama: {ID_1}
Date of registration: {DATE_1}
Mobile: {PHONE_1}
Email: {EMAIL_1}
City: {LOCATION_1}
Emergency contact: {PERSON_2} ({PHONE_2})
Fees paid: {AMOUNT_1}
Receipt number: {ORDER_1}""",
    "chat": """[10:02] {PERSON_1}: hey, did you send the form to {ORG_1}?
[10:03] {PERSON_2}: not yet, they asked for my ID number again
[10:03] {PERSON_2}: it's {ID_2} right? I always forget
[10:04] {PERSON_1}: no idea, check your wallet. also my new number is {PHONE_1}
[10:05] {PERSON_2}: saved. send me the address in {LOCATION_1} when you can
[10:06] {PERSON_1}: {ADDRESS_1}. see you at 7""",
    "hr_memo": """INTERNAL MEMO - CONFIDENTIAL

To: All department managers
From: {PERSON_1}, Human Resources
Date: {DATE_1}

Effective {DATE_2}, {PERSON_2} will join {ORG_1} as Finance Manager and will be based in {LOCATION_1}. {PERSON_2} previously worked at {ORG_2}.

Salary transfers will be made to IBAN {IBAN_2}. For payroll questions, contact {EMAIL_1} or extension 204.""",
    "invoice": """INVOICE {ORDER_1}

Billed to: {PERSON_1}
Company: {ORG_1}
Address: {ADDRESS_1}, {LOCATION_1}
Phone: {PHONE_1}

Description: Annual software subscription
Amount due: {AMOUNT_1}
Due date: {DATE_1}

Payment received by card {CARD_1}. Thank you for your business.
Questions? Write to {EMAIL_3}.""",
    "medical_appointment": """Appointment confirmation

Patient: {PERSON_1}
File number: {ID_1}
Clinic: {ORG_1}, {LOCATION_1}
Doctor: Dr. {PERSON_2}
Date: {DATE_1} at 10:30

Please arrive 15 minutes early. To reschedule, call {PHONE_3} or reply to {EMAIL_3}. The consultation fee is {AMOUNT_1}.""",
    "complaint": """To the customer service team of {ORG_1},

My name is {PERSON_1} and I am writing about order {ORDER_1}, placed on {DATE_1}. The package was delivered to the wrong address in {LOCATION_1} instead of {ADDRESS_1}.

I already spoke to your agent {PERSON_2}, who promised a refund of {AMOUNT_1} to my card {CARD_1}, but nothing has arrived. You can reach me on {PHONE_1} or at {EMAIL_1}.

Regards,
{PERSON_1}""",
    "certificate": """TO WHOM IT MAY CONCERN

This is to certify that {PERSON_1}, holder of ID number {ID_1}, completed a summer internship at {ORG_1} in {LOCATION_1} from {DATE_1} to {DATE_2}.

During this period, {PERSON_1} worked under the supervision of {PERSON_2} and showed excellent commitment.

For verification, please contact {EMAIL_2}.""",
    "meeting_notes": """Meeting notes - {DATE_1}

Attendees: {PERSON_1}, {PERSON_2}, {PERSON_3}

1. {PERSON_1} presented the quarterly numbers for {ORG_1}. Revenue grew by 12 percent.
2. {PERSON_2} will visit the client in {LOCATION_1} next week and share the contract draft.
3. {PERSON_3} asked everyone to send travel receipts to {EMAIL_3}.

Next meeting: {DATE_2}, room 204.""",
    "bank_notice": """Dear customer,

A transfer of {AMOUNT_1} was made on {DATE_1} from your account ending 4471 to {PERSON_1} (IBAN {IBAN_1}).

If you did not authorise this transfer, call {ORG_1} immediately on {PHONE_3} and mention reference {ORDER_1}. We will never ask for your full card number by email.""",
    "rental_agreement": """RENTAL AGREEMENT SUMMARY

Landlord: {PERSON_1}
Tenant: {PERSON_2}, ID {ID_2}
Property: {ADDRESS_1}, {LOCATION_1}
Monthly rent: {AMOUNT_1}
Start date: {DATE_1}

The tenant can be reached on {PHONE_2} and {EMAIL_2}. Rent is paid to IBAN {IBAN_1}.""",
    "no_pii_notice": """Library notice

The library will be closed for maintenance on {DATE_1}. All borrowed books are automatically extended by 7 days. Late fees of {AMOUNT_1} per day will resume on {DATE_2}.

Study rooms on the second floor remain open from 8:00 to 22:00. Thank you for your understanding.""",
}

_SLOT_PATTERN = re.compile(r"\{([A-Z]+)_(\d+)\}")


def fill_template(template: str, rng: random.Random) -> tuple[str, list[dict]]:
    """Fill one template and return ``(text, gold_entities)``.

    Each gold entity is ``{"label", "start", "end"}`` with offsets into the
    returned text.
    """
    values: dict[tuple[str, str], str] = {}
    parts: list[str] = []
    entities: list[dict] = []
    position = 0  # length of the output written so far
    cursor = 0  # how far we have read in the template

    for match in _SLOT_PATTERN.finditer(template):
        literal = template[cursor : match.start()]
        parts.append(literal)
        position += len(literal)

        slot, number = match.group(1), match.group(2)
        category, maker = SLOT_MAKERS[slot]
        if (slot, number) not in values:
            context = values.get(("PERSON", number))  # EMAIL_1 uses PERSON_1's name
            values[(slot, number)] = maker(rng, context)
        value = values[(slot, number)]

        if category:
            entities.append({"label": category, "start": position, "end": position + len(value)})
        parts.append(value)
        position += len(value)
        cursor = match.end()

    parts.append(template[cursor:])
    text = "".join(parts)
    if clean_text(text) != text:
        raise ValueError("Template output is not canonical text; offsets would shift.")
    return text, entities


def generate_documents(count: int, seed: int) -> list[dict]:
    """Generate ``count`` fictional documents with gold labels."""
    rng = random.Random(seed)
    template_names = list(TEMPLATES)
    documents = []
    for index in range(count):
        template_name = template_names[index % len(template_names)]
        text, entities = fill_template(TEMPLATES[template_name], rng)
        documents.append({"id": f"doc_{index + 1:03d}", "template": template_name, "text": text, "entities": entities})
    return documents


def text_to_pdf(text: str, path: Path, title: str = "") -> None:
    """Write text into a simple, text-based PDF (used for sample files)."""
    document = pymupdf.open()
    page_width, page_height, margin, line_height = 595, 842, 56, 15
    page = document.new_page(width=page_width, height=page_height)
    y = margin
    if title:
        page.insert_text((margin, y), title, fontsize=14, fontname="hebo")
        y += line_height * 2

    for paragraph in text.split("\n"):
        if paragraph == "\f":  # a form-feed line starts a new page
            page = document.new_page(width=page_width, height=page_height)
            y = margin
            continue
        lines = textwrap.wrap(paragraph, width=88) or [""]
        for line in lines:
            if y > page_height - margin:
                page = document.new_page(width=page_width, height=page_height)
                y = margin
            page.insert_text((margin, y), line, fontsize=10.5, fontname="helv")
            y += line_height
    document.set_metadata({"title": title, "author": "PrivacyLens synthetic generator"})
    document.save(path)
    document.close()


def make_scanned_pdf(path: Path) -> None:
    """Create a PDF that contains only an image (to test the scanned-PDF message)."""
    source = pymupdf.open()
    page = source.new_page()
    page.insert_text((72, 100), "This page will be turned into an image.", fontsize=14)
    pixmap = page.get_pixmap(dpi=70)
    source.close()

    document = pymupdf.open()
    page = document.new_page()
    page.insert_image(page.rect, pixmap=pixmap)
    document.save(path)
    document.close()


def build_evaluation_set(config: dict, count: int = 60, seed: int = 303) -> dict:
    """Generate documents, split them into dev/test and save everything.

    Output:
      * ``data/synthetic/dev/*.txt`` and ``data/synthetic/test/*.txt``
      * ``data/gold/ner_gold.json``   (PERSON, ORG, LOCATION, ADDRESS)
      * ``data/gold/regex_gold.json`` (EMAIL, PHONE, ID_LIKE, CREDIT_CARD, IBAN)
      * ``data/samples/`` example TXT/PDF files for trying the app
    """
    from src.schemas import NER_CATEGORIES, REGEX_CATEGORIES

    documents = generate_documents(count, seed)
    rng = random.Random(seed)
    order = list(range(count))
    rng.shuffle(order)
    dev_indexes = set(order[: count // 3])  # one third dev, two thirds test

    synthetic_dir = resolve_path(config["evaluation"]["synthetic_dir"])
    gold_dir = resolve_path(config["evaluation"]["gold_dir"])
    samples_dir = resolve_path("data/samples")
    for folder in (synthetic_dir / "dev", synthetic_dir / "test", gold_dir, samples_dir):
        folder.mkdir(parents=True, exist_ok=True)

    ner_gold, regex_gold = {}, {}
    for index, document in enumerate(documents):
        split = "dev" if index in dev_indexes else "test"
        (synthetic_dir / split / f"{document['id']}.txt").write_text(document["text"], encoding="utf-8")
        base = {"split": split, "template": document["template"]}
        ner_gold[document["id"]] = {**base, "entities": [e for e in document["entities"] if e["label"] in NER_CATEGORIES]}
        regex_gold[document["id"]] = {**base, "entities": [e for e in document["entities"] if e["label"] in REGEX_CATEGORIES]}

    for name, gold in (("ner_gold.json", ner_gold), ("regex_gold.json", regex_gold)):
        with open(gold_dir / name, "w", encoding="utf-8") as file:
            json.dump(gold, file, indent=1)

    # Sample files for the app demo (generated with a different seed, so they
    # are not part of the evaluation set).
    samples = {doc["template"]: doc for doc in generate_documents(len(TEMPLATES), seed + 1)}
    (samples_dir / "sample_email.txt").write_text(samples["email"]["text"], encoding="utf-8")
    text_to_pdf(samples["formal_letter"]["text"], samples_dir / "sample_letter.pdf", "Appointment Letter")
    text_to_pdf(samples["cv"]["text"], samples_dir / "sample_cv.pdf")
    text_to_pdf(
        # "\f" (form feed) puts every document on its own page.
        "\n\f\n".join(samples[name]["text"] for name in ("registration_form", "invoice", "rental_agreement", "hr_memo")),
        samples_dir / "sample_multi_page.pdf",
    )
    make_scanned_pdf(samples_dir / "sample_scanned.pdf")

    summary = {
        "documents": count,
        "dev_documents": len(dev_indexes),
        "test_documents": count - len(dev_indexes),
        "templates": len(TEMPLATES),
        "seed": seed,
        "entities_per_category": {},
    }
    for document in documents:
        for entity in document["entities"]:
            summary["entities_per_category"][entity["label"]] = summary["entities_per_category"].get(entity["label"], 0) + 1
    logger.info("Synthetic evaluation set created: %s", summary)
    return summary
