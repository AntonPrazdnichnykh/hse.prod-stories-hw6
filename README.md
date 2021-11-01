# Домашнее задание к лекции от 12.10.2021 курса "Истории из продакшена"

Список необходимых для работы spell checker'a модулей находится в файле `requirements.txt`

Примеры работы кода можно найти в ноутбуке `examples.ipynb`

В основе данного спэлчекера лежит Hunspell. Поддерживается только английский язык. Для каждого слова, которое не является
именем собсвенным и не содержит никаких символов, кроме букв Hunspell определяет, правильно ли
написанно данное слово. Если нет, то с помощью того же Hunspell генерирутеся список кандидатов, для каждого из них
считается расстояние Левенштейна, Жаро-Винклера и отрицательный логарифм вероятности тройки `<предыдущее слово> <кандиат> >следущее слово>` 
в модели, где каждое слово зависит только от предыдущего. Корпус взят [отсюда](https://www.norvig.com/ngrams/).
Лучший кандидат тот, у которого среднее этих фичей наименьшее. На него и менятеся рассматнриваемой слово. Все оствальные
символы исходого текста остаются неизменными.

