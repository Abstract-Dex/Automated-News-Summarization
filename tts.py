# import torch
# from parler_tts import ParlerTTSForConditionalGeneration
# from transformers import AutoTokenizer
# import soundfile as sf

# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# # device = "mps" if torch.backends.mps.is_available() else "cpu"

# model = ParlerTTSForConditionalGeneration.from_pretrained(
#     "ai4bharat/indic-parler-tts").to(device)
# tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
# description_tokenizer = AutoTokenizer.from_pretrained(
#     model.config.text_encoder._name_or_path)

# prompt = "শিঞ্জন একটা বোকাচোদা ছেলে "
# description = "A female speaker with an Indian accent delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."

# description_input_ids = description_tokenizer(
#     description, return_tensors="pt").to(device)
# prompt_input_ids = tokenizer(prompt, return_tensors="pt").to(device)

# generation = model.generate(input_ids=description_input_ids.input_ids, attention_mask=description_input_ids.attention_mask,
#                             prompt_input_ids=prompt_input_ids.input_ids, prompt_attention_mask=prompt_input_ids.attention_mask)
# audio_arr = generation.cpu().numpy().squeeze()
# sf.write("indic_tts_out.wav", audio_arr, model.config.sampling_rate)

# 1️⃣ Install kokoro

import soundfile as sf
from IPython.display import display, Audio
from kokoro import KPipeline
import spacy
spacy.load('en_core_web_sm')

languages = {
    "English": "a",
    "British English": "b",
    "Hindi": "h",
}

voices = {
    "English Male": "am_adam",
    "English Female": "af_heart",
    "British English Male": "bm_daniel",
    "British English Female": "bf_alice",
    "Hindi Female": "hf_alpha",
    "Hindi Male": "hm_omega",
}


def get_voice(lang, gender):
    lang_code = languages[lang]
    voice_code = voices[f"{lang} {gender}"]
    return lang_code, voice_code


def generate_audio(text, language, gender):
    lang, voice = get_voice(language, gender)
    pipeline = KPipeline(lang_code=lang)
    generator = pipeline(
        text, voice=voice, speed=1, split_pattern=r'\n+')
    for i, (gs, ps, audio) in enumerate(generator):
        # print(gs, ps)
        # display(Audio(data=audio, rate=24000, autoplay=i == 0))
        sf.write(f'tts_out.wav', audio, 24000)


generate_audio("""# Tesla Finalizes Showroom Locations in India

## Summary
Tesla has finalized locations for its showrooms in India, with sites selected in Delhi and Mumbai. These locations are part of Elon Musk's plan to enter the Indian electric vehicle (EV) market. The move marks a significant step toward establishing Tesla's presence in one of the world's fastest-growing automotive markets.

## Key Points
- Tesla has chosen a 3-story building in Connaught Place, Delhi, for its first showroom.
- A property in Mumbai's Worli area has also been finalized for another showroom.
- Both locations are in upscale areas, reflecting Tesla's strategy to target premium customers.
- Tesla is in discussions with the Indian government to secure tax incentives for EV imports.
- The company plans to initially import vehicles to India before considering local manufacturing.

## Conclusion
Tesla's decision to finalize showroom locations in Delhi and Mumbai underscores its commitment to entering the Indian EV market. With strategic locations in premium areas and ongoing discussions with the government, Tesla is positioning itself to make a strong impact in India's burgeoning electric vehicle sector.""", "English", "Male")
