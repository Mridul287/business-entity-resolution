"""
A 39-row sample of REAL rows, used by both the preprocessing and the blocking
tests.

Why this file exists
--------------------

`tests/conftest.py`'s data policy is that no test reads `dataset/`, because the
real TSVs are ~500 MB each and are not committed. That policy is right and it
has a hole in it for anything multilingual: the synthetic 14-row frame in
`tests/preprocessing/conftest.py` has two Devanagari rows and one Gujarati row,
which is enough to prove a function is not broken but nowhere near enough to
prove script *detection* is calibrated. If the codepoint ranges were off by a
block, the synthetic frame would still pass.

So these rows are copied out of the real TSVs verbatim, script by script, and
frozen here as literals. They are the same device `test_clean_text.py` already
uses for mojibake: the real signatures written down, rather than re-read at
test time. `origin` records which file each row came from so a surprising
result can be checked against the source.

The shape
--------

39 rows: 3 for each of the nine Indic scripts the profile found in the names
(27 rows), plus 3 mixed-script names that are Latin *and* something else, 8
plain-ASCII English names, and 1 accented-Latin French name. So 31 rows set the
flag and 8 do not -- a ratio chosen for balance, not because it resembles the
real 11-19%, which is a population property measured by
`python -m src.preprocessing.multilingual` against real data, not something 39
hand-picked rows can represent.

The per-row script expectations live in `tests/preprocessing/test_multilingual.py`
as literals rather than being derived here. Deriving them from `detect_scripts`
would make every assertion agree with the code under test by construction, which
is exactly the calibration check this file exists to provide.
"""
from __future__ import annotations

import pandas as pd

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]

# (business_name, business_address, country, origin, what the row is for)
SAMPLED_ROWS: list[tuple[str, str, str, str, str]] = [
    # --- Devanagari (Hindi/Marathi) --------------------------------------
    ("आनंद फाउंडेशन प्राइवेट लिमिटेड",
     "L-1318/38 GROUND FLOOR SANGAM VIHAR, DELHI", "India", "test_source2",
     "Devanagari name, ASCII address"),
    ("व्हाइट मीडिया एजेंसीज",
     "MUMBAI, Maharashtra, UNIQUE INDUSTRIAL ESTATE, OFF VEER SAVARKAR MARG, PRABHADEVI, MUMBAI, UNIT NO.-315, MUMBAI",
     "India", "test_source2", "Devanagari name, English Maharashtra in the address"),
    ("टेक सॉल्यूशंस बेकरी प्रा. लि.",
     "301-314, MUMBAI, T), Maharashtra", "India", "test_source2",
     "Devanagari name, unbalanced paren in the address"),
    # --- Gujarati ---------------------------------------------------------
    ("સિલ્વર એનર્જી લિમિટેડ",
     "HN 18 F 502 SIDDHI ELEGANCE 3 SAHJANAND NAGAR OPP FORTUNE NR RADHE HOTEL, RAJKOT, Gujarat",
     "India", "test_source2", "Gujarati name, English Gujarat in the address"),
    ("શિવ સુપર કન્સલ્ટન્સી લિમિટેડ",
     "AHMEDABAD, HN A-577 103, AHMEDABAD, Gujarat, ISCON PARK, OPP. STAR BAZAR, SATELLITE",
     "India", "test_source2", "Gujarati name"),
    ("પરફેક્ટ કન્સલ્ટિંગ ગારમેન્ટ્સ પ્રાઇવેટ લિમિટેડ",
     "DARSHANAM TRADE, CENTER-3, B/S DCP, TF-11, GANDHINAGAR, Gujarat, VADODARA",
     "India", "test_source2", "Gujarati name, mixed-case English Gujarat"),
    # --- Gurmukhi (Punjabi) ----------------------------------------------
    ("ਸਕਾਈ ਅਰਿਹੰਤ ਗਲੋਬਲ ਪ੍ਰਾ. ਲਿ.",
     "NO 3121, WARD NO. 9 NEEM WALA CHOWK, MOHALI, SAHIBZADA AJIT SINGH NAGAR, ਪੰਜਾਬ",
     "India", "test_source2", "Gurmukhi name AND native-script Punjab in the address"),
    ("ਲਕਸ਼ਮੀ ਐਨਰਜੀ ਪ੍ਰਾਈਵੇਟ ਲਿਮਟਿਡ",
     "4 F.F., LANE NO 2, BEHIND NECTOR LIFE SCIENCES VILL. SAIDPURA, DERA BASSI, PUNJAB, MOHALI, Punjab",
     "India", "test_source2", "Gurmukhi name, English Punjab twice"),
    ("ਵ੍ਹਾਈਟ ਪ੍ਰੋਪਰਟੀਜ਼ ਪ੍ਰਾ. ਲਿ.",
     "STREET NO. 4 NEW CANTT ROAD, FARIDKOT, GOLEWALA, Punjab", "India", "test_source2",
     "Gurmukhi name"),
    # --- Telugu -----------------------------------------------------------
    ("కృష్ణా ఇంపెక్స్ లిమిటెడ్",
     "H.NO 27TH FLOOR, Telangana, HYDERABAD, SERI LINGAMPALLY, G SQUARE, NEAR WELLS FARGO, RAIDURGAM",
     "India", "test_source2", "Telugu name, English Telangana in the address"),
    ("సుప్రీమ్ మోడర్న్ ఇన్వెస్ట్‌మెంట్ ప్రైవేట్ లిమిటెడ్",
     "PRAKASAM, Andhra Pradesh, MAIN ROAD, TRIPURANTHAKAM VILLAGE & MANDAL, 6-137A, TRIPURANTHAKAM",
     "India", "test_source2", "Telugu name carrying a ZWNJ (U+200C) mid-word"),
    ("అర్బన్ ఎనర్జీ ప్రైవేట్ లిమిటెడ్",
     "PLOT NO-10, HYDERABAD, TIRUMALAGIRI, Andhra Pradesh", "India", "test_source2",
     "Telugu name, English Andhra Pradesh"),
    # --- Tamil ------------------------------------------------------------
    ("யுனைடெட் சிஸ்டம்ஸ் கன்ஸ்ட்ரக்ஷன்ஸ் பிரைவேட் லிமிடெட்",
     "3/5 A BLOCK, EGMORE NUNGAMBAKKA, CHENNAI, Tamil Nadu", "India", "test_source2",
     "Tamil name"),
    ("ஆல்ஃபா புராஜெக்ட்ஸ் பிரைவேட் லிமிடெட்",
     "DOOR NO G-320 N, NSR ROAD SAIBABA COLONY, COIMBATORE, Tamil Nadu", "India",
     "test_source2", "Tamil name"),
    ("ஸ்கை சர்வீசஸ் பிரைவேட் லிமிடெட்",
     "NO 708 7TH FLOOR, SUITE NO.1149, SPENCER PLAZA MALL, ANNA SALAI, CHINTADRIPET, CHENNAI, CHENNAI, Tamil Nadu",
     "India", "test_source2", "Tamil name"),
    # --- Kannada ----------------------------------------------------------
    ("ಗುರು ಎಸ್ಟೇಟ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್",
     "NO.54/3, SUBBARAMA CHETTY ROAD BASAVANAGUDI, BANGALORE, BENGALURU, Karnataka",
     "India", "test_source2", "Kannada name"),
    ("ಸನ್ ವೆಂಚರ್ಸ್ ಟೆಕ್ನಾಲಜೀಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್",
     "4/13 K B H COLONY, BANGALORE SOUTH, Karnataka", "India", "test_source2",
     "Kannada name"),
    ("ಹೈ ಯುನೈಟೆಡ್ ಮಾರ್ಕೆಟಿಂಗ್",
     "H.NO 389 247, K.G.HALLI, BANGALORE, Karnataka", "India", "test_source2",
     "Kannada name"),
    # --- Malayalam --------------------------------------------------------
    ("അൽ കൺസ്ട്രക്ഷൻസ് ഫുഡ്സ് പ്രൈവറ്റ് ലിമിറ്റഡ്",
     "XVIII/C-37, FIRST FLOOR, U BROTHERS BUILDING, GURUVAYOOR ROAD, KUNNAMKULAM POST, THRISSUR, കേരളം",
     "India", "test_source2", "Malayalam name AND native-script Kerala in the address"),
    ("വൈറ്റ് ഇന്റർനാഷണൽ പ്രൈവറ്റ് ലിമിറ്റഡ്",
     "DOOR NO: 5/252 - A, OLD NO 6/156 CHAMBANNOOR KAVALA, ANGAMALY SOUTH, ANGAMALY, ERNAKULAM, Kerala",
     "India", "test_source2", "Malayalam name, ASCII Kerala"),
    ("ഗ്ലോബൽ ভാരത് ഫൗണ്ടേഷൻ ലിമിറ്റഡ്",
     "C/O RAGHUTHAMAN KALLIL KARANTHUR POST KUNNAMANGALAM, KOZHIKODE, Keralam", "India",
     "test_source2", "Malayalam name, Keralam (an alternate English spelling of Kerala)"),
    # --- Bengali ----------------------------------------------------------
    ("স্টার ইন্ডাস্ট্রিজ প্রাইভেট লিমিটেড",
     "DOOR NO 9/1 SOVARAM BYSACK STREET, KOLKATA, HOWRAH, West Bengal", "India",
     "test_source2", "Bengali name"),
    ("গ্যালাক্সি ফুড প্রাইভেট লিমিটেড",
     "36A/ SHYAMA PRASAD MUKHERJE ROAD, KOLKATA, HOWRAH, West Bengal", "India",
     "test_source2", "Bengali name"),
    ("গ্রীন কনসালট্যান্টস প্রাইভেট লিমিটেড",
     "71/1, BAIDYABATI, HOOGHLY, West Bengal", "India", "test_source2", "Bengali name"),
    # --- Oriya ------------------------------------------------------------
    ("ନ୍ୟୁ ବିଲଡର୍ସ୍ ପ୍ରାଇଭେଟ୍ ଲିମିଟେଡ୍",
     "DOOR NO 959 GKV-121, BHUBANESWAR, Odisha", "India", "test_source2",
     "Oriya name, ASCII Odisha"),
    ("ବିଜୟ କନଷ୍ଟ୍ରକସନ୍ସ୍ ପ୍ରାଇଭେଟ୍ ଲିମିଟେଡ୍",
     "PURI, Odisha, #1ST FLOOR, PURI", "India", "test_source2", "Oriya name"),
    ("ଶ୍ରୀ ଇଣ୍ଡଷ୍ଟ୍ରିଜ୍ ଫର୍ନିଚର୍ ପ୍ରାଇଭେଟ୍ ଲିମିଟେଡ୍",
     "NO 96 C/O NITYANANDA BEHERA, JAGATSINGHAPUR, JAGATSINGHPUR, Orissa", "India",
     "test_source2", "Oriya name, Orissa (an alternate English spelling of Odisha)"),
    # --- mixed script: Latin AND something else ---------------------------
    ("Shiva Big इंटरनेशनल Private Limited",
     "PLOT 0022 SC-318, SHASTRI NAGAR, Uttar Pradesh", "India", "test_source2",
     "Latin brand name + Devanagali legal form"),
    ("Eastern ইনফ্রা Private Limited",
     "89 K C DAS ROAD, P.O AND P.S-SANTIPUR, SANTIPUR, West Bengal", "India",
     "test_source2", "Latin + Bengali"),
    ("Shivam ಫೈನಾನ್ಸ್ LLP",
     "747(9/1), 47TH CROSS 1ST MAIN, 8TH BLOCK, JAYANAGAR, BANGALORE, Karnataka",
     "India", "test_source2", "Latin + Kannada"),
    # --- plain ASCII: the flag must be False for all of these -------------
    ("Apex Summit", "67 KENTUCKY ST, SALYERSVILLE, KY", "US", "test_source2", "plain ASCII US"),
    ("West Charter School Downtown LLC", "1462 SIXTH AVENUE, KANKAKEE, IL", "US",
     "test_source2", "plain ASCII US"),
    ("BEACON CORP", "7812 COLUSA ST, PO BOX 899, PORT ORCHARD, WA", "US", "test_source2",
     "plain ASCII US"),
    ("ARCE & BELTRAN PARTNERS", "5116 80TH STREET, STILLWATER, OK", "US", "test_source2",
     "plain ASCII US"),
    ("Vision Partners Corp", "IA, Iowa City, 1064 Newton Rd, Unit 11", "US", "test_source1",
     "plain ASCII, from S1"),
    ("Red Perfect Trading",
     "Mirzapur, Ews 12, Uttar Pradesh, Mirzapursadar, Awas Vikas Colony", "India",
     "test_source1", "ASCII Indian address, English Uttar Pradesh, from S1"),
    ("Nandlal Kisan LLP",
     "D-61 Ifs Apartmentmayur Vihar I, New Delhi, East Delhi, Delhi", "India",
     "test_source1", "ASCII, New Delhi, from S1"),
    ("Perfect Investments Private",
     "Plot No.99, Flat No.201, Sri Dhama Apts., Road No.4, Shaikpet, Hyderabad, TG",
     "India", "test_source3", "ASCII with an ISO state code, from S3"),
    # --- accented Latin: the flag is True, the strict reading is False ----
    # This is the row that separates has_non_latin_name from
    # has_indian_script_name, and the whole test_source1 2.35% share.
    ("SCI Ptit Àmicale", "18 RUE JEN ZAY, Dunkerque, Nord", "France", "test_source2",
     "accented Latin only: non-ASCII, but no transliteration risk"),
]

SAMPLE_ROWS: list[dict[str, str]] = [
    {
        "entity_id": f"SAMP-{i:03d}",
        "business_name": name,
        "business_address": address,
        "country": country,
    }
    for i, (name, address, country, _origin, _purpose) in enumerate(SAMPLED_ROWS, start=1)
]
SAMPLE_ORIGINS: dict[str, str] = {
    f"SAMP-{i:03d}": origin for i, (_n, _a, _c, origin, _p) in enumerate(SAMPLED_ROWS, start=1)
}
SAMPLE_ROW_COUNT = len(SAMPLE_ROWS)

# One row per entity_id, with its source file and its per-row script
# expectations filled in by the tests that need them.
sample_source_df = pd.DataFrame(SAMPLE_ROWS, columns=SOURCE_COLUMNS).astype(str)


def rows_with_script(script: str, column: str = "business_name") -> list[str]:
    """Entity ids in the sample whose `column` is written in `script`."""
    from src.preprocessing.multilingual import detect_scripts

    return [
        row["entity_id"]
        for row in SAMPLE_ROWS
        if script in detect_scripts(row[column])
    ]


def rows_without_any_script(column: str = "business_name") -> list[str]:
    """Entity ids in the sample whose `column` is pure ASCII."""
    from src.preprocessing.multilingual import detect_scripts

    return [
        row["entity_id"]
        for row in SAMPLE_ROWS
        if not detect_scripts(row[column])
    ]
