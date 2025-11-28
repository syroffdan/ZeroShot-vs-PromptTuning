import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import open_clip
from tqdm import tqdm
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import random
from collections import defaultdict
import logging
import sys
from contextlib import redirect_stdout, redirect_stderr
import io

# Настройка логирования
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(message)s',
#     handlers=[
#         logging.FileHandler('training.log', encoding='utf-8'),
#         logging.StreamHandler(sys.stdout)
#     ]
# )
#
# logger = logging.getLogger()
#
#
# class UniversalLogger:
#     """Логирует всё как INFO, включая tqdm"""
#
#     def __init__(self, logger):
#         self.logger = logger
#         self.terminal = sys.stdout
#
#     def write(self, message):
#         self.terminal.write(message)
#         if message.strip():
#             self.logger.info(message.strip())
#
#     def flush(self):
#         self.terminal.flush()
#
# # Перенаправляем оба потока в один логгер
# sys.stdout = UniversalLogger(logger)
# sys.stderr = sys.stdout
#
# # TQDM логгер
# def tqdm_write(message, file=None, end="\n"):
#     if file is not None:
#         # Для записи в файлы используем оригинальную функцию
#         original_tqdm_write(message, file=file, end=end)
#     else:
#         logger.info(message.strip())
#
# original_tqdm_write = tqdm.write
# tqdm.write = tqdm_write

# Задание сида для воспроизводимости
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Установлен сид {seed}")

# Конфиг
seed = 111
set_seed(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Устройство {gpu_name}")
else:
    print(f"Устройство {device}")

batch_size = 16
num_epochs = 50
lr = 0.5e-3
prompt_length = 12
val_split = 0.2
num_classes = 5
images_per_class = 50

# Выбираем 5 классов из CIFAR-10
selected_classes = ['airplane', 'automobile', 'bird', 'cat', 'dog']
class_names = selected_classes

# Создаем папку для сохранения графиков
os.makedirs('results', exist_ok=True)

# Загрузка модили Clip
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.to(device)

# Датасет
print("Загрузка датасета CIFAR-10...")

# Загружаем CIFAR-10
cifar10_dataset = CIFAR10(root='./data', train=True, download=True, transform=preprocess)

# Создаем маппинг выбранных классов
class_to_idx = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'dog': 4}
selected_class_indices = [class_to_idx[cls] for cls in selected_classes]

# Собираем индексы для каждого выбранного класса
class_indices = defaultdict(list)

for idx, (_, label) in enumerate(cifar10_dataset):
    if label in selected_class_indices:
        class_indices[label].append(idx)

# Создаем сбалансированный датасет
selected_indices = []
selected_final_labels = []

for class_idx in selected_class_indices:
    # Выбираем фиксированное количество изображений для каждого класса
    indices_for_class = class_indices[class_idx]
    selected_class_samples = np.random.choice(
        indices_for_class,
        images_per_class,
        replace=False
    )
    selected_indices.extend(selected_class_samples)
    selected_final_labels.extend(
        [selected_classes.index(selected_classes[selected_class_indices.index(class_idx)])] * images_per_class)

# Проверяем число изображений в классах
print("Число изображений в классах:")
for i, class_name in enumerate(selected_classes):
    count = selected_final_labels.count(i)
    print(f"  {class_name}: {count} изображений")

# Разделение на train/val
print(f"\nРазделение на train/val...")

def stratified_split(indices, labels, val_ratio=0.2):
    # Группируем индексы по классам
    class_to_indices = defaultdict(list)
    for idx, label in zip(indices, labels):
        class_to_indices[label].append(idx)

    train_indices = []
    val_indices = []

    # Для каждого класса делаем отдельное разделение на train/val
    for class_label, class_idx_list in class_to_indices.items():
        n_val = int(len(class_idx_list) * val_ratio)
        n_train = len(class_idx_list) - n_val
        # Перемешиваем индексы класса
        np.random.shuffle(class_idx_list)
        # Разделяем
        train_indices.extend(class_idx_list[:n_train])
        val_indices.extend(class_idx_list[n_train:n_train + n_val])
    return train_indices, val_indices

# Выполняем разделение
train_indices, val_indices = stratified_split(selected_indices, selected_final_labels, val_split)

# Создаем подмножества
train_subset = Subset(cifar10_dataset, train_indices)
val_subset = Subset(cifar10_dataset, val_indices)

# Кастомный датасет для правильных лейблов
class CIFAR5Dataset(Dataset):
    def __init__(self, subset, original_indices, all_labels):
        self.subset = subset
        # Создаем маппинг от индекса в subset к оригинальному лейблу
        self.labels = [all_labels[original_indices.index(idx)] for idx in subset.indices]

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, _ = self.subset[idx]
        label = self.labels[idx]
        return image, torch.tensor(label)


# Создаем финальные датасеты
train_dataset = CIFAR5Dataset(train_subset, selected_indices, selected_final_labels)
val_dataset = CIFAR5Dataset(val_subset, selected_indices, selected_final_labels)

# Проверяем распределение изображений после разделения
print("\nРаспределение изображений после разделения:")
train_class_counts = [0] * len(selected_classes)
val_class_counts = [0] * len(selected_classes)

for _, label in train_dataset:
    train_class_counts[label] += 1
for _, label in val_dataset:
    val_class_counts[label] += 1

for i, class_name in enumerate(selected_classes):
    print(f"  {class_name}: {train_class_counts[i]} train, {val_class_counts[i]} val")

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    generator=torch.Generator().manual_seed(seed)
)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"\n Распределение данных:")
print(f"   Всего: {len(selected_indices)} images")
print(f"   Train: {len(train_dataset)} изображений ({len(train_dataset) / len(selected_indices) * 100:.1f}%)")
print(f"   Val: {len(val_dataset)} изображений ({len(val_dataset) / len(selected_indices) * 100:.1f}%)")

# Zero-Shot CLIP
# Тестирование различных вариантов промптов
print("\nТестирование различных вариантов промптов CLIP Baselines...")

# Разные варианты текстовых промптов для тестирования
baseline_prompts_variants = {
    "simple": [
        "a photo of an airplane",
        "a photo of an automobile",
        "a photo of a bird",
        "a photo of a cat",
        "a photo of a dog"
    ],
    "detailed": [
        "a high quality photograph of an airplane flying in the sky",
        "a clear image of an automobile on the road",
        "a detailed picture of a bird with feathers",
        "a sharp photo of a cat with fur",
        "a bright image of a dog with tail"
    ],
    "class_only": [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "dog"
    ],
    "contextual": [
        "an airplane in the aviation context",
        "an automobile in transportation",
        "a bird in nature environment",
        "a cat as a domestic animal",
        "a dog as a pet"
    ],
    "artistic": [
        "a painting of an airplane",
        "a drawing of an automobile",
        "an artwork of a bird",
        "a sketch of a cat",
        "an illustration of a dog"
    ],
    "scientific": [
        "aircraft vehicle airplane",
        "motor vehicle automobile",
        "avian species bird",
        "feline animal cat",
        "canine animal dog"
    ]
}


def evaluate_zero_shot(loader, prompts, model, tokenizer, prompt_name="Baseline"):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    class_correct = [0] * len(prompts)
    class_total = [0] * len(prompts)

    with torch.no_grad():
        # Закодируем текстовые промпты один раз
        text_tokens = tokenizer(prompts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        for images, labels in tqdm(loader, desc=f"Zero-Shot {prompt_name}"):
            images, labels = images.to(device), labels.to(device)

            # Закодируем изображения
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Вычислим схожесть
            similarity = (100.0 * image_features @ text_features.T)
            _, preds = similarity.max(1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Считаем точность по классам
            for i in range(len(prompts)):
                class_mask = (labels == i)
                if class_mask.any():
                    class_correct[i] += (preds[class_mask] == labels[class_mask]).sum().item()
                    class_total[i] += class_mask.sum().item()

    accuracy = correct / total if total > 0 else 0
    class_accuracies = []
    for i in range(len(prompts)):
        if class_total[i] > 0:
            class_accuracies.append(class_correct[i] / class_total[i])
        else:
            class_accuracies.append(0.0)

    return accuracy, all_preds, all_labels, class_accuracies


# Тестируем все варианты промптов
baseline_results = {}

for prompt_name, prompts in baseline_prompts_variants.items():
    print(f"\nТестирование: {prompt_name}")
    print("Промпты:", prompts)

    accuracy, preds, labels, class_acc = evaluate_zero_shot(
        val_loader, prompts, model, tokenizer, prompt_name
    )

    baseline_results[prompt_name] = {
        'accuracy': accuracy,
        'predictions': preds,
        'labels': labels,
        'class_accuracies': class_acc,
        'prompts': prompts
    }

    print(f"{prompt_name} accuracy: {accuracy:.4f}")

    # Выводим точность по классам
    for i, class_name in enumerate(selected_classes):
        print(f"   {class_name}: {class_acc[i]:.4f}")

# Находим лучший baseline
best_baseline = max(baseline_results.items(), key=lambda x: x[1]['accuracy'])
best_baseline_name = best_baseline[0]
best_baseline_acc = best_baseline[1]['accuracy']

print(f"\nЛучший Baseline: '{best_baseline_name}' с точностью: {best_baseline_acc:.4f}")

# Визуализация сравнения baseline промптов
plt.figure(figsize=(12, 8))

# График сравнения accuracy
plt.subplot(2, 1, 1)
prompt_names = list(baseline_results.keys())
accuracies = [baseline_results[name]['accuracy'] for name in prompt_names]
colors = ['skyblue' if name != best_baseline_name else 'gold' for name in prompt_names]

bars = plt.bar(prompt_names, accuracies, color=colors, alpha=0.7)
plt.title('Zero-Shot CLIP Performance with Different Prompts')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Добавляем значения на столбцы
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'{acc:.4f}', ha='center', va='bottom')

# Heatmap точности по классам
plt.subplot(2, 1, 2)
class_acc_matrix = np.array([baseline_results[name]['class_accuracies'] for name in prompt_names])
sns.heatmap(class_acc_matrix,
            annot=True,
            fmt='.3f',
            xticklabels=selected_classes,
            yticklabels=prompt_names,
            cmap='YlOrRd')
plt.title('Class-wise Accuracy for Different Prompts')
plt.xlabel('Classes')
plt.ylabel('Prompt Types')

plt.tight_layout()
plt.savefig('results/baseline_prompts_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Сохраняем лучший baseline для дальнейшего сравнения
best_baseline_data = baseline_results[best_baseline_name]
zero_shot_val_acc = best_baseline_data['accuracy']
zero_shot_preds = best_baseline_data['predictions']
zero_shot_labels = best_baseline_data['labels']

print(f"\nИспользуется лучший baseline '{best_baseline_name}' для сравнения с prompt tuning")

# Prompt tuning
embedding_dim = model.text_projection.shape[1]

# Отдельные софт промпты для каждого класса
torch.manual_seed(seed)
learned_prompts = nn.Parameter(
    torch.randn(len(class_names), prompt_length, embedding_dim, device=device)
)

# Базовые эмбеддинги классов (замороженные)
text_tokens = tokenizer(class_names).to(device)
with torch.no_grad():
    class_embeddings = model.encode_text(text_tokens)

def apply_prompt_tuning(class_embeddings, learned_prompts):
    tuned_embeddings = []
    for i, emb in enumerate(class_embeddings):
        tuned = torch.cat([learned_prompts[i], emb.unsqueeze(0)], dim=0)
        tuned = tuned.mean(0)
        tuned_embeddings.append(tuned)
    tuned_embeddings = torch.stack(tuned_embeddings, dim=0)
    tuned_embeddings = tuned_embeddings / tuned_embeddings.norm(dim=-1, keepdim=True)
    return tuned_embeddings

# Вычисляем веса для балансировки классов
class_counts = [len(train_dataset) // num_classes] * num_classes
total_count = sum(class_counts)
weights = [total_count / c for c in class_counts]
weights = torch.tensor(weights, dtype=torch.float32, device=device)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam([learned_prompts], lr=lr)

# Цикл обучения
print("\nЗапуск Prompt Tuning...")

train_losses = []
train_accuracies = []
val_accuracies = []
all_features = []
all_labels_tsne = []

for epoch in range(num_epochs):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        tuned_embeddings = apply_prompt_tuning(class_embeddings, learned_prompts)
        logits = 100.0 * image_features @ tuned_embeddings.T

        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = logits.max(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=loss.item(), acc=correct / total)

    # Сохраняем метрики обучения
    epoch_loss = total_loss / total
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    # Валидация
    val_correct, val_total = 0, 0
    val_preds = []
    val_true = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            tuned_embeddings = apply_prompt_tuning(class_embeddings, learned_prompts)
            logits = 100.0 * image_features @ tuned_embeddings.T
            _, preds = logits.max(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            val_preds.extend(preds.cpu().numpy())
            val_true.extend(labels.cpu().numpy())

            # Сохраняем фичи для t-SNE на последней эпохе
            if epoch == num_epochs - 1:
                all_features.extend(image_features.cpu().numpy())
                all_labels_tsne.extend(labels.cpu().numpy())

    val_acc = val_correct / val_total if val_total > 0 else 0
    val_accuracies.append(val_acc)

    print(
        f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | Val Acc: {val_acc:.4f}")

# Визуализация

# Графики обучения
plt.figure(figsize=(15, 5))

# Loss
plt.subplot(1, 3, 1)
plt.plot(train_losses, 'b-', label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# Accuracy
plt.subplot(1, 3, 2)
plt.plot(train_accuracies, 'g-', label='Training Accuracy')
plt.plot(val_accuracies, 'r-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()

# Сравнение методов
plt.subplot(1, 3, 3)
methods = ['Zero-Shot', 'Prompt-Tuned']
accuracies = [zero_shot_val_acc, val_acc]
colors = ['red', 'green']
bars = plt.bar(methods, accuracies, color=colors, alpha=0.7)
plt.title('Сравнение методов')
plt.ylabel('Accuracy')
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'{acc:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('results/training_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

# Матрица ошибок для prompt tuning
plt.figure(figsize=(10, 8))
cm = confusion_matrix(val_true, val_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Матрица ошибок - Prompt Tuning')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('results/confusion_matrix_prompt_tuning.png', dpi=300, bbox_inches='tight')
plt.show()

# Матрица ошибок для zero-shot
plt.figure(figsize=(10, 8))
cm_zero_shot = confusion_matrix(zero_shot_labels, zero_shot_preds)
sns.heatmap(cm_zero_shot, annot=True, fmt='d', cmap='Reds',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Матрица ошибок - Zero-Shot')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('results/confusion_matrix_zero_shot.png', dpi=300, bbox_inches='tight')
plt.show()

# t-SNE визуализация
if len(all_features) > 0:
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_features) - 1))
    features_2d = tsne.fit_transform(np.array(all_features))

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                          c=all_labels_tsne, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, ticks=range(len(class_names)))
    plt.clim(-0.5, len(class_names) - 0.5)
    plt.title('t-SNE Visualization of Image Features (Prompt-Tuned)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    # Добавляем легенду
    handles, _ = scatter.legend_elements()
    plt.legend(handles, class_names, title="Classes")

    plt.savefig('results/tsne_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

# Вывод информации о классификации
print("\nPrompt Tuning:")
print(classification_report(val_true, val_preds, target_names=class_names))

print("Zero-Shot:")
print(classification_report(zero_shot_labels, zero_shot_preds, target_names=class_names))

# Итоговые результаты
print("Итоговые результаты")

# Сравниваем все методы
methods_data = []

# Добавляем все baseline промпты
for prompt_name, result in baseline_results.items():
    methods_data.append({
        'method': f'Zero-Shot ({prompt_name})',
        'accuracy': result['accuracy'],
        'type': 'zero_shot'
    })

# Добавляем prompt tuning
methods_data.append({
    'method': 'Prompt Tuning',
    'accuracy': val_acc,
    'type': 'prompt_tuning'
})

# Создаем DataFrame для удобного отображения
comparison_df = pd.DataFrame(methods_data)
comparison_df = comparison_df.sort_values('accuracy', ascending=False)

print("\nОценка методов:")
print(comparison_df.to_string(index=False))

# Визуализация сравнения всех методов
plt.figure(figsize=(14, 8))

# График сравнения всех методов
colors = ['lightblue' if x == 'zero_shot' else 'lightgreen' for x in comparison_df['type']]
bars = plt.bar(comparison_df['method'], comparison_df['accuracy'], color=colors, alpha=0.7)

plt.title('Сравнение методов: Zero-Shot vs Prompt Tuning', fontsize=14)
plt.ylabel('Accuracy', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)

# Добавляем значения на столбцы
for bar, acc in zip(bars, comparison_df['accuracy']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{acc:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('results/all_methods_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Результаты
print(f"\nРезультаты:")
print(f"Best Zero-Shot: '{best_baseline_name}' - {best_baseline_acc:.4f}")
print(f"Prompt-Tuned: {val_acc:.4f}")
print(f"Абсолютное улучшение: {(val_acc - best_baseline_acc):+.4f}")
print(f"Относительное улучшение: {((val_acc - best_baseline_acc) / best_baseline_acc * 100):+.2f}%")

# Сохранение результатов
torch.save({
    "prompts": learned_prompts.detach().cpu(),
    "class_names": class_names,
    "baseline_results": baseline_results,
    "best_baseline": {
        'name': best_baseline_name,
        'accuracy': best_baseline_acc,
        'prompts': best_baseline_data['prompts']
    },
    "final_accuracy": val_acc,
    "training_history": {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies
    },
    "config": {
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "lr": lr,
        "prompt_length": prompt_length,
        "images_per_class": images_per_class,
        "val_split": val_split,
        "seed": seed,
        "split_type": "stratified_per_class"
    }
}, "cifar5_learned_prompts.pth")

print("Все результаты и промпты сохраняются в cifar5_learned_prompts.pth")
print("Все визуализации сохраняются 'results/'")