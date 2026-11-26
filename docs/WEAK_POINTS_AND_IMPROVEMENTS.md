# Weak Points and Improvements

This file lists, honestly, where PrivacyLens is weak and what would make it better.
It is meant for the team: to know the limits before the presentation, to answer questions, and
to choose what to improve next.

How to read the tables:

- **Impact** says how much the point matters for a user or for the grade: High, Medium or Low.
- **How we know** says whether the point was *measured* (a number from the evaluation),
  *tested* (tried by hand on 16 or 26 November 2026), or is *by design* (a known choice, not tested).
- Numbers are for model version 2 (26 November 2026) unless version 1 is named.

## 1. The ten most important points

| # | Weak point | Impact | Section |
|---|---|---|---|
| 1 | Organisation names that are not companies (universities, hospitals) are often missed | High | 2.1 |
| 2 | Text written in lower case or in CAPITALS is handled badly | High | 2.2 |
| 3 | Arabic text is not analysed | High | 2.3 |
| 4 | Several kinds of sensitive data are not detected at all (dates of birth, passwords, passport numbers, web addresses) | High | 2.4 |
| 5 | All tests use synthetic documents; the real-world quality is unknown | High | 4.1 |
| 6 | Most of the code was written by an AI assistant; the team must be able to explain it and report it truthfully | High | 7.1 |
| 7 | Any long number is flagged as an ID, which gives false alarms and pushes the risk score up | Medium | 3.1 |
| 8 | The risk score reaches 100 too easily, so it does not separate documents well | Medium | 5.2 |
| 9 | Scanned PDFs and text inside pictures are not analysed | Medium | 5.1 |
| 10 | Complex PDF layouts (several columns, tables, rotated pages) were not tested | Medium | 5.1 |

## 2. The AI model (names, organisations, places, addresses)

### 2.1 Organisations that are not companies

**How we know: measured and tested.** On the 40 fictional test documents model version 2 finds
only 57% of organisation names exactly (precision 0.80, recall 0.57, F1 0.67; version 1: 0.81,
0.62, 0.70). On the public test set, where organisations are companies, it reaches F1 0.93
(version 1: 0.92).

**What the team's extra ORG documents changed (26 November 2026).** Version 2 was trained with 350
of the team's business documents. It now recognises company names in short business forms very
well (ORG F1 1.00 on held-out templates, version 1: 0.81) and is slightly better on the public test
set, but the problem below is **not solved**: those documents contain only company names, in 20
sentence patterns. Details: the Dataset Update document.

Examples tested by hand (version 1 on 16 November, both versions on 26 November):

| Text | Version 1 | Version 2 |
|---|---|---|
| "She studied at Imam Abdulrahman Bin Faisal University and works at King Fahd Hospital in Dammam." | Only "Dammam" is found. Both institutions are missed. | The same, and "King Fahd" is wrongly marked as an address. |
| "Intern at Red Sea Analytics" (in the sample CV of the app) | Only "Analytics". | "Sea Analytics" (still incomplete). |
| "Al Waha Medical Center" | Missed (or only "Al Waha", as a place). | Missed. |
| "Oasis Travel Agency" | Only "Oasis" is marked. | **Found completely.** |

**Cause.** The training data labels company names. Institutions follow another pattern.

**Improvements.**

- Add institution names to the training data: documents with universities, hospitals, ministries
  and schools, written in **many different sentence patterns**, with every person, place and
  organisation labelled. More copies of the same templates do not help (the team's documents
  showed this: using them three times gave no gain).
- Try Few-NERD, which the team found: its organisation types (company, education, government,
  ...) give the variety that is missing. Check its licence first.
- Add a simple rule: when a detected name is directly followed by a word such as University,
  Hospital, Center, Academy, Agency or Ministry, extend the item to include it.
- Check the result on the development documents, not on the test documents
  (`scripts/06_compare_models.py --stage validation`).

### 2.2 Lower-case and upper-case text

**How we know: tested.**

| Text | Result |
|---|---|
| "please contact ahmed alqahtani or sara alharbi at the riyadh office." | Nothing is found. |
| "APPLICANT: KHALID AL-MUTAIRI. EMPLOYER: GULF HORIZON TRADING COMPANY, JEDDAH." | The name is split into "KHALID AL" and "MUTAIRI" (the hyphen stays visible), "GULF HORIZON" is called a person, the rest of the company name and "JEDDAH" are missed. |

**Cause.** The model uses capital letters as a signal, and the training documents use normal
capitalisation. Chat messages (lower case) and form headings (capitals) are common in real life.

**Improvement.** Add lower-case and upper-case copies of some training documents and retrain.
This needs no new data and about 20 more minutes of training.

### 2.3 Arabic text

**How we know: tested.** In "الاسم: محمد العتيبي، المدينة: الرياض، الجوال: 0551234567" only the
phone number is found. Names and cities written in Arabic are not detected.

**Cause.** The base model and the dataset are English only. This was already outside the scope
of Plan v2.

**Improvement (larger).** Use a multilingual base model and add Arabic training data with a clear
licence. The app should warn the user when a document contains mostly Arabic text; today it
gives no such warning.

### 2.4 Kinds of data that are not detected

**How we know: tested.** These were tried and none was detected:

| Not detected | Example tried |
|---|---|
| Date of birth | "DOB: 14/03/2001" |
| Password | "Password: Tr!ckyPass9" |
| IP address | "192.168.1.44" |
| Web address | "https://example.com/u/noura.h" |
| Passport number with a letter | "P12345678" |
| Licence plate | "ABC 1234" |
| User name | "noura_h99" |
| Postal code | "34428" after an address |

**Improvement (good value for the effort).** The dataset already contains labels for most of
these (`date_of_birth` 30,787 examples, `password` 15,264, `ipv4` 15,020, `url` 82,599,
`user_name` 28,639, `license_plate` 11,211, `postcode` 14,307). To add a category: add it to
`label_map` in the configuration, to the category list in `src/schemas.py`, to the risk weights
and to the colours, then retrain and re-evaluate. Web and IP addresses can also be done with rules.

### 2.5 Training choices that were kept simple

| Point | Detail | Possible improvement |
|---|---|---|
| Only part of the data was used | 20,000 of about 200,000 documents | Train on 50,000 or more and compare |
| One training run | One random seed, 3 epochs, no search for better settings. Two runs with almost the same data (26 November) differed by 0.002 in validation F1, so differences of that size between versions can be chance. | Run 2 or 3 seeds and report the spread |
| Validation and training share "sister" documents | Nemotron-PII has a US and an international version of every document. Script 01 splits by row, so 420 of the 2,000 validation documents have their other version in the training split. Validation scores may be slightly optimistic; the test split is not affected. | Split by `uid` in `src/data_preparation.py` (`transform`) at the next rebuild |
| Small base model | DistilBERT was chosen for speed | Try a larger model if a stronger GPU is available |
| Results are not exactly repeatable | GPU training gives slightly different numbers each run | State this in the report; keep the trained model file |

## 3. The rules (emails, phones, IDs, cards, IBANs)

### 3.1 Any long number becomes an "ID"

**How we know: tested.** In "Order number 4829173650 was shipped. Invoice 2026001234, tracking
771234567890." all three numbers are flagged as `ID_LIKE`.

This is by design: for redaction, a false alarm is safer than a miss, and the user can untick it.
The cost is that ID-like numbers carry the highest risk weight (30 points for the first one,
up to 60 for several), so an ordinary invoice can look far more risky than it is.

**Improvements.** Give generic long numbers a lower risk weight than numbers that match a known
ID format (for example the 10-digit Saudi format). Use nearby words ("ID", "Iqama", "National
ID", "Order", "Invoice") to raise or lower confidence.

### 3.2 Other rule limits

| Point | How we know | Improvement |
|---|---|---|
| IDs that contain letters (passport "P12345678") are missed | Tested | Add rules for common passport and residence-permit formats |
| 16 digits written as 4-4-4-4 are always called a bank card, even with a wrong checksum | By design (the dataset's card numbers are invented and fail the checksum) | Show "checksum failed" in the table so the user can judge |
| Some phone formats are still missed | Measured: phone F1 0.94 on the public test set | Study the misses on the validation data and add formats |
| The IBAN and ID rules could not be tested on the public dataset | By design: the dataset has no matching labels. They were tested only on our 40 fictional documents (9 IBANs, 14 IDs). | Add more varied fictional examples |
| The street address keeps its final comma, and the postal code is left out | Tested | Trim punctuation; add the postal code to ADDRESS |

## 4. How the system was tested

### 4.1 Everything was tested on synthetic documents

**By design.** The project rule "no real personal data" means that no real document was used.
This is the right choice for privacy, but the real-world quality is therefore unknown.

- The public test set (F1 0.961) comes from the same generator as the training data.
  It shows that the model learned its task. It probably overstates quality on real documents.
- The team's ORG test documents (ORG F1 1.00) are short, regular forms whose company-name
  families were seen in training; this number is optimistic too.
- The 40 fictional test documents (F1 0.950) have a different style but come from only
  14 templates, written by the same developer who wrote the rules. With so few documents, one or
  two errors change a category's score visibly.

**Improvement.** Team members write 20 to 30 realistic documents by hand (invented people,
different styles, some lower case, some with institutions) and label them. This is the
cross-labelling that Plan v2 asked for.

### 4.2 Other points about the evaluation

| Point | Detail |
|---|---|
| Two rule fixes were made after reading test errors | The fictional-test scores for PHONE and CREDIT_CARD (1.00) are slightly optimistic. With model version 1 the strictly held-out result was F1 0.939 overall instead of 0.949. This is recorded in the project log. |
| The origin of the team's ORG documents is not recorded | Who wrote the 500 documents and with which tool must be added to the data-sources document (marked `TO COMPLETE`). |
| The correct answers for the fictional documents were not checked by a person | They are written automatically by the generator. A member should review a sample. |
| The comparison with the baseline model is not fully fair | The baseline was trained with other labelling habits and has no ADDRESS category. Part of the gap (0.687 against 0.961) comes from that. Say this when presenting the number. |
| The confidence threshold could be higher | On the development documents, 0.7 gives better precision with the same recall. 0.5 was kept to stay on the safe side. |
| No test with real users | Nobody outside the build has tried the app yet. |

## 5. PDF handling and the app

### 5.1 PDF limits

| Point | How we know | Improvement |
|---|---|---|
| Scanned PDFs are refused; text inside pictures is not analysed (the app shows a warning when a PDF contains pictures) | By design | Add OCR (reading text from images), for example with Tesseract |
| Bookmarks, attachments, form fields, comments and hidden layers are not inspected | By design | Remove or check them before saving |
| Complex layouts were not tested: several columns, tables, rotated pages, right-to-left text | Not tested | Test with such PDFs and fix what fails |
| A missed item stays visible | By design: redaction is only as good as detection | Let the user select extra text by hand in the app |
| Only PDF and TXT are accepted | By design | Add Word (DOCX) files |
| The PDF library is fixed at an older version | Tested: version 2.1 moves text next to a redaction bar | Re-test with each new version; a test in the project detects the bug |
| Limits: 20 MB, 200 pages, 500,000 characters | By design (in the configuration) | Raise after testing speed and memory |

### 5.2 App and usability

| Point | How we know | Improvement |
|---|---|---|
| The risk score reaches 100 easily | Tested: the one-page sample CV with 10 items already scores 100 / 100 "Critical" | Lower the weights or change the caps so the score spreads over the range |
| The app needs Python and a setup of about 8 minutes | Tested with a new environment | Publish it online (then documents leave the user's computer, which must be explained), or build an installer |
| Rule detections show "None" in the Confidence column of the review table | Seen in the app | Show an empty cell or "rule" instead |
| One document at a time | By design | Allow several files in one scan |
| The interface is in English only | By design | Add Arabic labels |
| Speed without a graphics card is fine for normal documents | Measured: 0.2 s for 4 pages, about 4 s for 20 pages of text, about 30 s for 200,000 characters | Show a progress bar for long documents |

## 6. Engineering and reproducibility

### 6.1 Repository and storage

| Point | Detail | What to do |
|---|---|---|
| The model is stored with Git LFS | Each clone downloads the 250 MB model file. GitHub counts this against the repository owner's free Git LFS allowance (10 GB storage and 10 GB downloads per month on GitHub Free), and every committed model version stays in storage. | Commit a retrained model only when it is better. If the allowance runs out, upload the model to the Hugging Face Hub and write its name in `config/config.yaml` (`ner.hub_model`). |
| A clone without Git LFS gets a 1 KB placeholder instead of the model | The app then falls back to the weaker public model | Run `git lfs install` and `git lfs pull` (explained in the run guide) |
| The project folder is inside OneDrive | About 550 MB of dataset and model files are synchronised to the cloud | Move the project outside OneDrive, or exclude `data/raw` and `models` from sync |

### 6.2 Other engineering points

| Point | Detail |
|---|---|
| Very recent library versions | The fixed versions need Python 3.12 or newer. The installation was tested on Windows 11 only, not on macOS or Linux. |
| Tests that need the model are skipped when it is missing | All tests can pass on a computer without the trained model. Check that no test was skipped before trusting a green result. |
| The app is one file of about 620 lines | It works, but it is the hardest file to read. It could be split into one file per tab. |
| The app screens are not tested automatically | The logic behind them is tested (122 tests); the screens were checked by hand and with screenshots. |
| One licence is stricter than the others | PyMuPDF uses the AGPL licence. Fine for a course project with public code; check before any commercial use. |

## 7. Project and course points

### 7.1 Use of AI assistance

The first complete version of the code, the training run, the evaluation and the documentation
were produced by an AI assistant in one working session, from the team's plan. The course asks
for transparency about this.

- Every member should read `docs/CODE_GUIDE.md` and be able to explain the files they present.
- `AI_USAGE_REPORT.md` describes the AI work truthfully. The sections on what each member
  checked, changed and wrote are marked `TO COMPLETE` and must be filled in by the team itself.
- Do not present AI-written code as written independently.

### 7.2 Differences from Plan v2 that the team should confirm

| Plan v2 | What was built |
|---|---|
| No redacted PDF in this version | A redacted PDF is produced |
| Use a pretrained model | A pretrained model was fine-tuned (trained further) |
| Six categories | Nine: ADDRESS, CREDIT_CARD and IBAN were added |
| Members label the test documents and cross-check | Labels are generated automatically |

### 7.3 Deliverables still open

- The report as PDF: `docs/REPORT_DRAFT.md` is a draft that needs names, contributions and
  screenshots (screenshots of the app are now in `reports/screenshots`).
- The presentation slides.
- The team sections of the AI usage report.
- Topic registration in the course sheet, if not done.

## 8. Suggested order of work

**Quick (about an hour each)**

1. Give every team member (and the instructor, if the repository is private) access to the GitHub repository.
2. Fill in the team sections of the AI usage report, and record who made the 500 ORG documents and how.
3. Show an empty cell instead of "None" for rule detections.
4. Lower the risk weight of generic long numbers.

**Medium (half a day each)**

5. Retrain with lower-case and upper-case copies of training documents (point 2.2).
6. Add institution names (varied sentences, or Few-NERD) to the training data and the
   "University / Hospital" rule (point 2.1).
7. Add two or three new categories from the dataset, for example date of birth and web address (point 2.4).
8. Write and label 20 to 30 realistic test documents by hand (point 4.1).

**Larger (several days)**

9. Arabic support.
10. OCR for scanned documents.
11. Word (DOCX) input and manual selection of extra text to redact.

After every change to the model or the rules: run `scripts/04_evaluate.py` again, tune only on
the development documents, and write the change in `docs/PROJECT_LOG.md`.
