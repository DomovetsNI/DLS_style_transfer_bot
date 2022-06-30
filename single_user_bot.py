import telebot

from telebot import types

from config import TOKEN
from style_trans_nn import style_transfer

bot = telebot.TeleBot(TOKEN)

kb = types.InlineKeyboardMarkup(row_width=1)
key1 = types.InlineKeyboardButton(text="Пример переноса стиля",
                                  callback_data="sample")
key2 = types.InlineKeyboardButton(text="Сделать одну фотографию в стиле другой",
                                  callback_data="transfer")
key3 = types.InlineKeyboardButton(text="Посмотреть информацию по боту",
                                  callback_data="help")
kb.add(key1, key2, key3)


kb2 = types.InlineKeyboardMarkup(row_width=1)
content_key = types.InlineKeyboardButton(text="Основное изображение",
                                         callback_data="content")
style_key = types.InlineKeyboardButton(text="Переносимый стиль",
                                       callback_data="style")
kb2.add(content_key, style_key)

content_flag = 0
style_flag = 0


@bot.message_handler(commands=["start"])
def start(message):
    bot.send_message(message.chat.id,
                     "Привет! Это бот который позволяет переносить" +
                     "стили одной фотографии на другую!",
                     reply_markup=kb)


@bot.callback_query_handler(func=lambda c: c.data)
def reaction(callback):
    global content_flag, style_flag
    if callback.data == "help":
        bot.send_message(callback.message.chat.id,
                         "Этот бот был написан в качестве проекта на курсе DLS." +
                         "Он позволяет переносить стиль одной фотографии на другую.")
        bot.send_message(callback.message.chat.id,
                         "Список действий:",
                         reply_markup=kb)
    if callback.data == "sample":
        bot.send_photo(callback.message.chat.id,
                       photo=open('./sample/content.jpeg', 'rb'),
                       caption="Изображение на которое мы будем накладывать стиль")
        bot.send_photo(callback.message.chat.id,
                       photo=open('./sample/style.jpeg', 'rb'),
                       caption="Переносимый стиль")
        bot.send_photo(callback.message.chat.id,
                       photo=open('./sample/gen.jpeg', 'rb'),
                       caption="Полученное изображение")
        bot.send_message(callback.message.chat.id,
                         "Список действий:",
                         reply_markup=kb)
    if callback.data == "transfer":
        bot.send_message(callback.message.chat.id,
                         "Выберите какую фотографию будете прикреплять," +
                         "а затем прикрепите ее",
                         reply_markup=kb2)
    if callback.data == "content":
        content_flag = 1
    if callback.data == "style":
        style_flag = 1


@bot.message_handler(content_types="photo")
def get_image(message):
    global content_flag, style_flag, content_filename, style_filename
    file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    if content_flag == 1:
        content_filename = './content/' + message.photo[1].file_id + ".png"
        with open(content_filename, 'wb') as new_file:
            new_file.write(downloaded_file)
        content_flag = 2
    elif style_flag == 1:
        style_filename = './style/' + message.photo[1].file_id + ".png"
        with open(style_filename, 'wb') as new_file:
            new_file.write(downloaded_file)
        style_flag = 2
    if (content_flag == 2) and (style_flag == 2):
        bot.send_message(message.chat.id,
                         "Подождите, пожалуйста, пару минут пока картинка сгенерируется")
        gen_filename = style_transfer(content_filename, style_filename)
        bot.send_photo(message.chat.id,
                       photo=open(gen_filename, 'rb'),
                       caption="Вот и сгенерированная картинка готова!")
        content_flag, style_flag = 0, 0
        bot.send_message(message.chat.id,
                         "Вот весь функционал, который предоставляет бот",
                         reply_markup=kb)
    else:
        bot.send_message(message.chat.id,
                         "Выберите какую фотографию будете прикреплять," +
                         "а затем прикрепите ее",
                         reply_markup=kb2)
        return


if __name__ == "__main__":
    bot.polling()
