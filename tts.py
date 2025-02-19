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

# 🇪🇸 'e' => Spanish es
# 🇫🇷 'f' => French fr-fr
# 🇮🇳 'h' => Hindi hi
# 🇮🇹 'i' => Italian it
# 🇧🇷 'p' => Brazilian Portuguese pt-br
# 3️⃣ Initalize a pipeline
# 🇺🇸 'a' => American English, 🇬🇧 'b' => British English
# 🇯🇵 'j' => Japanese: pip install misaki[ja]
# 🇨🇳 'z' => Mandarin Chinese: pip install misaki[zh]
pipeline = KPipeline(lang_code='b')  # <= make sure lang_code matches voice

text = "Would u like a bottle of water"

generator = pipeline(
    text, voice='bm_daniel',  # <= change voice here
    speed=1, split_pattern=r'\n+'
)
for i, (gs, ps, audio) in enumerate(generator):
    # print(i)  # i => index
    # print(gs)  # gs => graphemes/text
    # print(ps)  # ps => phonemes
    # display(Audio(data=audio, rate=24000, autoplay=i == 0))
    sf.write(f'{i}.wav', audio, 24000)  # save each audio file
