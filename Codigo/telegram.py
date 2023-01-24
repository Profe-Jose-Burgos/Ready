import telebot
import random
import numpy as np
from backend import bag_of_words, model, words, labels, data

bot = telebot.TeleBot('5937779519:AAHQpSMYurGkcjw4-M_LPeJS6lDdgfsTBIE')

@bot.message_handler(func=lambda message: True)
def handle_message(message):
   # Procesar el mensaje y determinar la intención del usuario
    message_text = message.text  # obtenemos el texto del mensaje
    # creamos la representacion numerica del mensaje
    message_representation = bag_of_words(message_text, words)
    # hacemos la prediccion con el modelo entrenado
    result = model.predict([message_representation])
    # obtenemos el indice de la intencion con mayor probabilidad
    intent_index = np.argmax(result)
    if intent_index != None:
        intent = labels[intent_index]  # obtenemos la intencion correspondiente
        # Seleccionar una respuesta apropiada
        for intents in data['intents']:
            if intents['tag'] == intent:
                responses = intents['responses']
        # Seleccionar una respuesta al azar
        response = random.choice(responses)
    else:
        response = "Lo siento, no entendí tu mensaje. ¿Podrías aclararlo?"
    # Enviar respuesta al usuario
    bot.send_message(chat_id=message.chat.id, text=response)


bot.polling()
