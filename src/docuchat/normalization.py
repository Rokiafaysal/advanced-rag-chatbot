import re

TATWEEL = "\u0640"                              # ـ  (stretching, purely visual)
DIACRITICS = re.compile(r"[\u064B-\u0652\u0670]")  # ً ٌ ٍ َ ُ ِ ّ ْ  and superscript alef
ALEF_FORMS = re.compile(r"[إأآٱ]")


def normalize_arabic(text:str)->str:
    text=DIACRITICS.sub("", text)
    text = text.replace(TATWEEL, "")
    text = ALEF_FORMS.sub("ا", text)
    text = text.replace("ى", "ي")
    return text


