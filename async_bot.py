import logging

from aiogram import Bot, types
from aiogram.types import ChatActions, InputFile, ParseMode
from aiogram.types.message import ContentType
from aiogram.utils import executor
from aiogram.utils.markdown import text
from aiogram.dispatcher import Dispatcher

from config import TOKEN
from style_trans_nn import style_transfer

logging.basicConfig(format=u'%(filename)s [ LINE:%(lineno)+3s ]#%(levelname)+8s [%(asctime)s]  %(message)s',
                    level=logging.INFO)

one_photo_flg = 1

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)


@dp.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    await message.reply('Привет!\nЯ могу переносить стиль одной фотографии',
                        'на другую! \nИспользуй /help,'
                        'чтобы узнать, что я умею.')


@dp.message_handler(commands=['help'])
async def process_help_command(message: types.Message):
    msg = text('Для переноса стиля нужно прислать две фотографии:',
               'Первая -- это та, у которой будет изменен стиль,',
               'Вторая -- это та, в каком стиле все будет.', sep='\n')
    await message.reply(msg, parse_mode=ParseMode.MARKDOWN)


@dp.message_handler(content_types=['photo'])
async def process_photo_command(message: types.Message):
    global one_photo_flg

    content_path = './content/' + str(message.from_user.id) + '.png'
    style_path = './style/' + str(message.from_user.id) + '.png'

    if one_photo_flg == 1:
        one_photo_flg = 0
        await message.photo[-1].download(destination_file=content_path)
        message_text = 'Сохранил эту фотографию. Теперь скидывай стиль!'
        await bot.send_message(message.from_user.id, message_text)
        return

    one_photo_flg = 1
    await message.photo[-1].download(destination_file=style_path)

    await bot.send_chat_action(message.from_user.id,
                               ChatActions.UPLOAD_DOCUMENT)
    gen_path = style_transfer(content_path, style_path)

    message_text = 'Вот это фоточка!\nСохранил ее. Теперь скидывай вторую со стилем.'
    await bot.send_photo(message.from_user.id, InputFile(gen_path))


@dp.message_handler()
async def echo_message(msg: types.Message):
    await bot.send_message(msg.from_user.id, msg.text)


@dp.message_handler(content_types=ContentType.ANY)
async def unknown_message(msg: types.Message):
    message_text = text('Я не знаю, что с этим делать:(',
                        'Есть команда /help чтобы узнать возможности бота')
    await msg.reply(message_text, parse_mode=ParseMode.MARKDOWN)


if __name__ == '__main__':
    executor.start_polling(dp)
