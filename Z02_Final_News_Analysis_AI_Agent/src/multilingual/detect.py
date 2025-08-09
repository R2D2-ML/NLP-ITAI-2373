from langdetect import detect
from deep_translator import GoogleTranslator

def translate_text(text):
    translated = GoogleTranslator(source='auto', target='en').translate(text)
    print(translated)

    return translated

def language_check(text):
    # 1. Detect language
    lang = detect(text)
    print(f"Lang is: {lang}")

    # 2. Translate if not in user’s preferred language
    if lang != 'en':
        translated_text = translate_text(text)
        return translated_text

    return text

