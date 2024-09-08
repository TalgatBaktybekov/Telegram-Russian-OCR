import os
import logging
import cv2
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.utils import executor
from predict import TranscribeImage  
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import StatesGroup, State
import cv2 

logging.basicConfig(level=logging.INFO)


bot = Bot(token=os.environ['TGTOKEN'])
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

class Form(StatesGroup):
    waiting_for_photo = State()
    choosing_language = State()

@dp.message_handler(commands=['start'])
async def start_command(message: types.Message):
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    keyboard.add(types.KeyboardButton('Русский'))
    keyboard.add(types.KeyboardButton('English'))
    await message.answer(
        """
        Выберите язык для распознавания рукописного текста:
        Choose the language for handwritten text recognition:""",
        reply_markup=keyboard
    )
    await Form.choosing_language.set()

@dp.message_handler(state=Form.choosing_language, text='Русский')
async def set_language_russian(message: types.Message, state: FSMContext):
    await state.update_data(language='russian')
    await message.answer(
        """
    Отлично! Пожалуйста, отправьте изображение рукописного текста, следуя этим рекомендациям:
    1. Изображение должно быть хорошо освещено.
    2. Текст должен быть написан четко и без излишних украшений.
    3. Избегайте фотографий с нечетким или наклонным текстом.
    4. Убедитесь, что текст не выходит за пределы изображения и не искажен.""")
    await Form.waiting_for_photo.set()

@dp.message_handler(state=Form.choosing_language, text='English')
async def set_language_english(message: types.Message, state: FSMContext):
    await state.update_data(language='english')
    await message.answer("""
    Great! Please send the handwritten text image, following these guidelines:
    1. The image should be well-lit.
    2. The text should be written clearly and without excessive decorations.
    3. Avoid images with unclear or slanted text.
    4. Ensure the text is not cut off or distorted in the image.""")
    await Form.waiting_for_photo.set()

@dp.message_handler(content_types=['photo'], state=Form.waiting_for_photo)
async def handle_photo(message: types.Message, state: FSMContext):

    user_data = await state.get_data()

    language = user_data.get('language', 'russian')

    await message.answer(
        "Изображение получено! Пожалуйста подождите, это может занять до нескольких минут." 
        if language == 'russian' 
        else "Image received! Please wait, it may take a few minutes."
    )

    file_id = message.photo[-1].file_id
    file_info = await bot.get_file(file_id)
    file_path = file_info.file_path
    

    downloaded_file = await bot.download_file(file_path)
    
    directory = 'tg_images'
    files = os.listdir(directory)
    image_id = len([file for file in files if file.lower().endswith('.jpg')])

    image_path = os.path.join(directory, str(image_id) + 'id' + '.jpg')

    with open(image_path, 'wb') as new_file:
        new_file.write(downloaded_file.getvalue())

    image = cv2.imread(image_path)

    text, chunked_rows = TranscribeImage(image)

    word_save_dir = os.path.join(directory, str(image_id) + 'idChunks')
    os.makedirs(word_save_dir)

    count = 0

    for row in chunked_rows:
        for word in row:
            cv2.imwrite(word_save_dir + '/' + str(count) + '.jpg', word)
            count += 1
    
    log_user_request(message.from_user.id, image_path, text)

    await message.answer(text)
    
    await state.finish()

import csv

def log_user_request(user_id, image_path, text):

    csv_file_path = 'requests.csv'
    

    file_exists = os.path.isfile(csv_file_path)
    
    with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(['user_id', 'image_path', 'text'])

        writer.writerow([user_id, image_path, text])

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)

