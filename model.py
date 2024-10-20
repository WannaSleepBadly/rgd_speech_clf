import pymorphy3
import re
from torch import no_grad
import os
import psutil

from vosk import Model, KaldiRecognizer
from transformers import BertForSequenceClassification, BertTokenizer

import librosa
import numpy as np
import wave
import soundfile as sf

model = Model("weights/vosk-model-small-ru-0.22")
recognizer = KaldiRecognizer(model, 16000)

morph = pymorphy3.MorphAnalyzer(lang='ru')

clf_path = 'weights/rubert_tiny_noised'
classifier = BertForSequenceClassification.from_pretrained(clf_path)
tokenizer = BertTokenizer.from_pretrained(clf_path)


def classify(text):
    """
    Классификация текстовых транскрибций команд

    Args:
        text (str): Текстовая транскрибция команды

    Returns:
        int:  Предсказанный label
    """

    tokenized_text = tokenizer([text], return_tensors='pt')

    with no_grad():
        outputs = classifier(**tokenized_text)

    prediction = np.argmax(outputs.logits)

    return int(prediction)


def text_to_num(words):
    """
    Преобразует текстовое представление числительного на русском языке в численное значение

    Args:
        words (list[str]): Список слов из транскрибции

    Returns:
        int: Численное значение числительного
    """

    num_dict = {
        "ноль": 0,
        "один": 1,
        "два": 2,
        "три": 3,
        "четыре": 4,
        "пять": 5,
        "шесть": 6,
        "семь": 7,
        "восемь": 8,
        "девять": 9,
        "десять": 10,
        "одиннадцать": 11,
        "двенадцать": 12,
        "тринадцать": 13,
        "четырнадцать": 14,
        "пятнадцать": 15,
        "шестнадцать": 16,
        "семнадцать": 17,
        "восемнадцать": 18,
        "девятнадцать": 19,
        "двадцать": 20,
        "тридцать": 30,
        "сорок": 40,
        "пятьдесят": 50,
        "шестьдесят": 60,
        "семьдесят": 70,
        "восемьдесят": 80,
        "девяносто": 90,
        "сто": 100
    }

    num = 0
    for word in words:
        if word in num_dict:
            num += num_dict[word]

    return num


def get_num(line):
    """
    Извлечение числительных и преобразование в числовой формат

    Args:
        line (str): Текстовая транскрибция голосовой команды

    Returns:
        int: Численное значение атрибута или -1, если не нашлось числительных
    """

    words = line.split(' ')

    # morph = pymorphy2.MorphAnalyzer(lang='ru')
    nums = [word for word in words if 'NUMR' in morph.parse(word)[0].tag]  # Отбор числительных

    if nums:
        return text_to_num(nums)
    return -1


def get_attribute(text, label):
    """
    Извлечение атрибутов

    Args:
        text(str): Текстовая транскрибция аудио
        label (int): Предсказанный класс

    Returns:
        int: Численное значение атрибута или -1, если не нашлось числительных или у команды не предусмотрен атрибут
    """

    if label in [4, 10]:  # Только в командах 4 и 10 нужно извлекать количество вагонов
        return get_num(text)
    else:
        return -1


def preprocess(text):
    """
    Лемматизация текста

    Args:
        text(str): Текстовая транскрибция аудио

    Returns:
        str: Строка, состоящая из лемматизированных слов
    """
    text = text.replace('ё', 'е')
    text = re.sub(r'[^а-яА-ЯёЁ\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return ' '.join([morph.parse(word)[0].normal_form for word in text.split(' ')])


# Функция для оценки шума (по первым 0.5 секундам)
def estimate_noise(y, sr, noise_duration=0.5):
    """
    Оценивает уровень шума в аудиосигнале на основе первых нескольких секунд.

    Args:
        y (np.ndarray): Аудиосигнал в виде массива чисел (например, от librosa).
        sr (int): Частота дискретизации аудиосигнала.
        noise_duration (float): Длительность сегмента для оценки шума в секундах. По умолчанию 0.5.

    Returns:
        np.ndarray: Среднее значение спектра шума, рассчитанное на основе STFT.
    """
    noise_samples = int(noise_duration * sr)
    noise_part = y[:noise_samples]
    stft_noise = np.abs(librosa.stft(noise_part))
    return np.mean(stft_noise, axis=1)


# Применение спектрального вычитания
def spectral_subtraction(y, sr, noise_est):
    """
    Применяет спектральное вычитание для удаления шума из аудиосигнала.

    Args:
        y (np.ndarray): Аудиосигнал в виде массива чисел.
        sr (int): Частота дискретизации аудиосигнала.
        noise_est (np.ndarray): Оценка спектра шума.

    Returns:
        np.ndarray: Очищенный аудиосигнал после применения спектрального вычитания.
    """
    stft_speech = librosa.stft(y)
    magnitude_speech = np.abs(stft_speech)
    phase_speech = np.angle(stft_speech)

    # Вычитание спектра шума из спектра сигнала
    magnitude_clean = np.maximum(magnitude_speech - noise_est[:, np.newaxis], 0)

    # Восстановление сигнала с очищенным спектром
    stft_clean = magnitude_clean * np.exp(1j * phase_speech)
    y_clean = librosa.istft(stft_clean)

    return y_clean


def filter_noise(input_file, output_file):
    """
    Фильтрует шум из аудиофайла, используя спектральное вычитание и сохраняет очищенный файл.

    Args:
        input_file (str): Путь к входному аудиофайлу.
        output_file (str): Путь для сохранения очищенного аудиофайла.
    """

    data, samplerate = librosa.load(input_file, sr=None)
    noise_estimation = estimate_noise(data, samplerate)
    cleaned_data = spectral_subtraction(data, samplerate, noise_estimation)
    sf.write(output_file, cleaned_data, samplerate)


def check_and_convert_sample_rate(input_file, target_sample_rate=16000):
    """
    Проверяет и при необходимости изменяет частоту дискретизации аудиофайла.

    Args:
        input_file (str): Путь к входному аудиофайлу.
        target_sample_rate (int): Целевая частота дискретизации в Герцах. По умолчанию 16000.

    Returns:
        str: Путь к аудиофайлу с нужной частотой дискретизации.
    """
    data, samplerate = librosa.load(input_file, sr=None)
    if samplerate != target_sample_rate:
        output_file = "converted_audio.wav"

        # Команда ffmpeg для изменения sample rate
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                input_file,
                "-ar",
                str(target_sample_rate),
                output_file,
                "-y",
            ]
        )

        return output_file
    else:
        return input_file


def trans_one_audio(file_path):
    """
    Транскрибирует аудиофайл и извлекает текст.

    Args:
        file_path (str): Путь к аудиофайлу для транскрибирования.

    Returns:
        str: Текст, полученный в результате транскрибирования аудиофайла.
    """
    global recognizer
    pid = os.getpid()
    python_process = psutil.Process(pid)
    memoryUse = python_process.memory_info()[0] / 2.0**30

    # Если ОЗУ вышла за порог, инициализируем новый recognizer
    if memoryUse > 0.95:
        recognizer = KaldiRecognizer(model, 16000)

    temp_cleaned_file = "cleaned_audio.wav"
    file_path = check_and_convert_sample_rate(file_path)
    filter_noise(file_path, temp_cleaned_file)

    with wave.open(temp_cleaned_file, "rb") as wf:
        full_text = ""

        while True:
            data = wf.readframes(4000)  # Читаем 4000 фреймов за раз
            if len(data) == 0:
                break
            if recognizer.AcceptWaveform(data):
                result = recognizer.Result()
                # Извлекаем текст и добавляем его в полный текст
                full_text += eval(result)["text"] + " "
            else:
                # Можно игнорировать частичные результаты или использовать их
                recognizer.PartialResult()

        # Финальный результат
        final_result = recognizer.FinalResult()
        full_text += eval(final_result)["text"]

    return full_text.strip()


def form_answer(audio_file):
    """
    Получение выходов модели

    Args:
        self (str): Путь к аудио файлу

    Returns:
        dict: Словарь с предсказаниями
    """

    res = {'text': trans_one_audio(audio_file)}
    res['label'] = classify(res['text'])
    prep_text = preprocess(res['text'])
    res['attribute'] = get_attribute(prep_text, res['label'])

    return res
