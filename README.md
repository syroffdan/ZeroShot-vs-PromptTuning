# Нулевой‑шот и prompt‑tuning с CLIP
Сравнение методов нулевого обучения (zero-shot) и настройки промптов (prompt tuning) с использованием модели CLIP на датасете CIFAR-10.
## Описание
Этот проект сравнивает:
- **Zero-shot** CLIP с различными текстовыми шаблонами (prompts)
- **Prompt tuning** - тонкую настройку текстовых промптов на основе 5 классов CIFAR-10 (по 50 изображений на класс)
## Быстрый старт
### Клонирование репозитория
```bash
git clone https://github.com/syroffdan/ZeroShot-vs-PromptTuning.git
```
### Установка зависимостей
```bash
pip install torch torchvision open_clip-torch tqdm numpy matplotlib seaborn scikit-learn pandas
```
### Запуск программы
```bash
python neuro_shots.py
```
## Результаты

| Метод | Точность | Тип  |
|-------|----------|------|
| Prompt Tuning | 0.94 | prompt_tuning |
| Zero-Shot (artistic) | 0.86 | zero_shot |
| Zero-Shot (simple) | 0.84 | zero_shot |
| Zero-Shot (class_only) | 0.84 | zero_shot |
| Zero-Shot (detailed) | 0.84 | zero_shot |
| Zero-Shot (scientific) | 0.84 | zero_shot |
| Zero-Shot (contextual) | 0.76 | zero_shot |
