import asyncio
from googletrans import Translator


class Translator:
    def __init__(self):
        self.translator = Translator()

    async def __aenter__(self):
        return self.translator

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.translator.client.close()

    @staticmethod
    async def translate(text, dest='en'):
        async with Translator() as translator:
            result = await translator.translate(text, dest=dest)
            return result.text







# async def translate():
#         async with Translator() as translator:
#             result = await translator.translate('সর্বশেষ সংবাদ অনুযায়ী, ভারতের পশ্চিমবঙ্গের রানিগঞ্জে ১৩০ ফুট গভীর একটি পরিত্যক্ত কয়লা খনিতে পড়ে যাওয়া এক যুবকের মৃতদেহ উদ্ধার করা হয়েছে। উদ্ধারকারীদের প্রচেষ্টা সত্ত্বেও তাকে জীবিত উদ্ধার করা সম্ভব হয়নি।', dest='en')
#             print(result.text)

#             result = await translator.translate('According to the latest news, the body of a young man has been recovered after falling into an abandoned coal mine 130 feet deep in Raniganj, West Bengal, India. Despite the efforts of the rescuers, he could not be rescued alive.', dest='bn')
#             print(result.text)

#             with open('output.txt', 'w') as f:
#                 f.write(result.text)
#                 f.close()
