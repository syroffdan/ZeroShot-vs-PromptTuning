# Нулевой‑шот и prompt‑tuning с CLIP
Сравнение методов нулевого обучения (zero-shot) и настройки промптов (prompt tuning) с использованием модели CLIP на датасете CIFAR-10.
## Описание
Этот проект сравнивает:
- **Zero-shot** CLIP с различными текстовыми шаблонами (prompts)
- **Prompt tuning** - тонкую настройку текстовых промптов на основе 5 классов CIFAR-10 (по 50 изображений на класс)
## Быстрый старт
### Установка зависимостей
```bash
pip install torch torchvision open_clip-torch tqdm numpy matplotlib seaborn scikit-learn pandas
```
### Запуск программы
```bash
python neuro_shots.py
```
