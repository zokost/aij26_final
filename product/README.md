# Цифровой детектив

Сервис выделяет подозрительные области на фотографии. Оператор рассматривает маску, добавляет ручные отметки, комментарий и решение. По запросу OpenRouter/Together даёт текстовую подсказку по выбранной области.

Все команды выполняются **из корня этой папки**: после клонирования репозитория выполните `cd product`. Комплект самостоятельный: код обучения и датасеты для запуска не нужны. Основной способ запуска — Docker на Linux с NVIDIA GPU.

## Структура

```text
.
├── README.md
├── Dockerfile
├── compose.yaml
├── requirements.txt         # API и детектор; установка через uv pip install
├── requirements-mcp.txt     # отдельное окружение MCP
├── .env.example             # настройки без настоящего ключа
├── app/                     # API, обработка маски, пояснения, MCP
├── web/                     # интерфейс и его ресурсы
├── src/                     # архитектура, инференс и общие преобразования
├── weights/                 # создать и скачать артефакты перед сборкой
├── docs/
│   ├── API.md               # HTTP API, сценарий, лимиты
│   └── MCP.md               # подключение агента локально и по SSH
├── scripts/install_host.sh  # подготовка Ubuntu
├── tests/                   # smoke API и тесты MCP
└── SHA256SUMS               # контрольные суммы комплекта
```

## 1. Требования к Linux

- Linux x86_64; для установщика — Ubuntu 22.04/24.04.
- NVIDIA GPU от 8 ГБ, RAM от 16 ГБ; для VM и Docker-сборки рекомендуется диск 130 ГБ.
- Работающий драйвер NVIDIA с поддержкой CUDA 12.4 или новее.
- Docker Engine, Compose v2, NVIDIA Container Toolkit; интернет для первой сборки и OpenRouter.

Для Vast нужен шаблон **Ubuntu VM**, чтобы собирать и запускать Docker внутри машины.

```bash
nvidia-smi
sudo bash scripts/install_host.sh
docker version
docker compose version
nvidia-ctk --version
uv --version
```

Установщик ставит недостающие компоненты и **перезапускает Docker**. Запускайте его до других контейнеров. Драйвер NVIDIA должен быть установлен заранее. При `Driver/library version mismatch` дождитесь окончания обновления драйвера и согласуйте модуль ядра с библиотеками — обычно перезагрузкой VM. Пересборка приложения это не исправляет.

## 2. Куда положить модель

Веса хранятся отдельно [на Hugging Face](https://huggingface.co/misiskovo/aij26-final-weights/tree/main/weights/121) и не входят в Git. Скачайте три связанных артефакта **до Docker-сборки** по путям относительно корня `product/`:

```text
weights/121_NC17_e32/model.pt
weights/121_NC17_e32/policy.pkl
weights/121_NC17_e32/encoder_transform.pkl
```

`model.pt` — NC17 epoch32, SegFormer MiT-B2/decoder320. Два `.pkl` — замороженные ExtraTrees и scaler/PCA; без них рецепт №121 не воспроизводится. Размер checkpoint — 99 877 063 байта. Команды для Bash:

```bash
mkdir -p weights/121_NC17_e32
WEIGHTS_URL='https://huggingface.co/misiskovo/aij26-final-weights/resolve/main/weights/121'
HF_AUTH=()
if [ -n "${HF_TOKEN:-}" ]; then HF_AUTH=(-H "Authorization: Bearer $HF_TOKEN"); fi
for name in model.pt policy.pkl encoder_transform.pkl; do
  curl --fail --location --retry 3 "${HF_AUTH[@]}" "$WEIGHTS_URL/$name" \
    --output "weights/121_NC17_e32/$name"
done
printf '%s\n' \
  '40841b9c867d5c6f1ed8fcfacefb14787a377d58f770da86e67067f1d57a4cf1  weights/121_NC17_e32/model.pt' \
  '0bf2ba074b3609aefea2eba383ab512f5d635311315c76a79e3b08472023de25  weights/121_NC17_e32/policy.pkl' \
  'fc865212594a3e98eb3e9b1465d75225782a422bebc636f4a7f99ebc949b9758  weights/121_NC17_e32/encoder_transform.pkl' | sha256sum --check
unset HF_AUTH
```

Если Hugging Face отвечает `401/403`, перед скачиванием задайте токен аккаунта с доступом к репозиторию: `read -rsp 'Hugging Face token: ' HF_TOKEN; printf '\n'`. Затем повторите блок скачивания. Токен Hugging Face и ключ OpenRouter — разные ключи; не записывайте их в Git. Docker-сборка и запуск дополнительно проверяют SHA256 всех трёх артефактов.

Проверка исходных файлов сервиса; веса проверяются отдельным блоком выше:

```bash
sha256sum --check SHA256SUMS
```

Инференс: RGB → OpenCV INTER_AREA 640×640 → ImageNet-нормализация → FP32 → билинейное увеличение логитов до исходного H×W → sigmoid. Без TTA и удаления компонент. Базовый порог — 0.675. Если базовая маска занимает меньше 1% кадра, она сохраняется. Иначе 276 признаков карты, RGB и того же forward энкодера подаются замороженному калибратору: выбранный порог 0.5/0.6/0.65/0.675 может только расширить маску.

В сервисе один кадр за проход, без CUDA Graph/compile. Исходная посылка использовала batch4; FP32 batch1 может отличаться отдельными граничными пикселями. В `/health` указан базовый порог, в `model.threshold` ответа `/detect` — фактически выбранный. Не задавайте `DETECTOR_THRESHOLD`, `DETECTOR_FAST=1` или другую геометрию для №121: калибратор зафиксирован под этот рецепт.

Источники: [артефакты](https://huggingface.co/misiskovo/aij26-final-weights/tree/main/weights/121), [код инференса](https://github.com/zokost/aij26_final/tree/c734d565eed23cd32573051bd4136e9a8bd85777/stage2). Параметры и метод выбора не переобучались; сериализованные артефакты загружаются с проверкой SHA256 и списком разрешённых NumPy/sklearn-типов.

## 3. Настроить ключ OpenRouter

```bash
umask 077
cp --no-clobber .env.example .env
chmod 600 .env
```

Откройте `.env` любым редактором и задайте:

```dotenv
DETECTIVE_PORT=8080
DETECTIVE_BIND=127.0.0.1
OPENROUTER_API_KEY=ВАШ_КЛЮЧ_OPENROUTER
OPENROUTER_EXPLANATIONS_ENABLED=1
OPENROUTER_EXPLANATION_MODEL=deepseek/deepseek-v4.1-flash-1017
```

Вместо `ВАШ_КЛЮЧ_OPENROUTER` вставьте действующий ключ без пробелов. Настоящего ключа в комплекте нет. `.env` исключён из Git и Docker-контекста. Не добавляйте ключ в Dockerfile, браузер или MCP-конфигурацию. Администратор Docker имеет доступ к окружению контейнера.

Для работы без внешних пояснений оставьте ключ пустым и `OPENROUTER_EXPLANATIONS_ENABLED=0`: детектор продолжит работать. Для пояснений нужны баланс и доступ к модели в [OpenRouter](https://openrouter.ai/settings/keys). Провайдер Together закреплён в коде, reasoning и fallback выключены.

## 4. Запустить через Dockerfile

Выберите один способ запуска, чтобы не занять порт дважды.

**Через Compose:** он сам собирает образ из Dockerfile и читает `.env`.

```bash
docker compose up --build -d --wait --wait-timeout 240
curl --fail http://127.0.0.1:8080/health
docker compose logs --tail 50 detective
```

**Без Compose:** тот же Dockerfile, ключ передаётся во время запуска контейнера.

```bash
docker build -t aij-detective:stage2-121 .
docker run -d --name aij-detective --restart unless-stopped --init \
  --gpus 1 --env-file .env \
  -p 127.0.0.1:8080:8765 aij-detective:stage2-121
curl --fail http://127.0.0.1:8080/health
docker logs --tail 50 aij-detective
```

Внутренний порт контейнера — 8765. Внешний порт прямого `docker run` задаётся `-p`, независимо от `DETECTIVE_PORT` в `.env`. В Compose внешний порт берётся из `.env`.

В `/health` ожидаются `status: ok`, `mode: inference`, `device: cuda`, правильный SHA256 и `explanations.available: true`, если пояснения включены. Последнее означает наличие настройки; реальный доступ к провайдеру проверяется кнопкой пояснения. Первый инференс может быть медленнее.

Проверка передачи GPU:

```bash
docker run --rm --gpus 1 aij-detective:stage2-121 nvidia-smi
```

Для удалённого сервера откройте **на своём компьютере** туннель:

```bash
ssh -N -p SSH_PORT USER@SERVER -L 8080:127.0.0.1:8080
```

Интерфейс: <http://127.0.0.1:8080/>. Туннель должен оставаться открытым. Сервис по умолчанию доступен только через localhost; авторизации и квот API нет. Для публичного доступа нужны отдельные HTTPS и контроль доступа.

### Замена ключа и остановка

Измените `.env`, затем пересоздайте контейнер:

```bash
docker compose up -d --force-recreate --wait
docker compose down
```

Для прямого `docker run`: `docker stop aij-detective`, затем `docker rm aij-detective` и повтор команды запуска. Обычный `restart` не меняет ключ. Если раньше экспортировали настройки OpenRouter в оболочке, сначала выполните `unset OPENROUTER_API_KEY OPENROUTER_EXPLANATIONS_ENABLED OPENROUTER_EXPLANATION_MODEL`: окружение оболочки имеет приоритет над `.env` в Compose. Не публикуйте вывод `docker inspect` / `docker compose config` с секретами.

## 5. Запуск Python без Docker через uv

Для Linux с NVIDIA GPU. Это отдельный способ запуска; остановите Docker-сервис, если он занимает порт 8080.

```bash
uv python install 3.11
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip check
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available()"
set -a
source .env
set +a
export OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1
python -m uvicorn app.service:app --host 127.0.0.1 --port 8080 --workers 1
```

`requirements.txt` содержит API, PyTorch и зависимости модели. Docker использует этот же файл; базовый образ PyTorch 2.6.0 / CUDA 12.4 закреплён digest. Прямые версии пакетов закреплены, полного lock-файла транзитивных зависимостей нет. `uvicorn` сам не читает `.env`, поэтому перед Python-запуском она экспортируется в окружение. Остановка — `Ctrl+C`.

MCP устанавливается **в другое окружение** из `requirements-mcp.txt`: его SDK требует другую версию Pydantic. Установка и подключение — [docs/MCP.md](docs/MCP.md).

## Работа и проверка

Загрузите фото → выберите область → рассмотрите её с масштабом и без маски → при необходимости запросите пояснение → добавьте отметки и решение → сохраните проверку.

Доступны очередь фотографий, ручная разметка, полное сохранение/открытие, обращения с несколькими материалами и отчётом, отдельный исходник для сравнения, короткое видео с временной шкалой и MCP. Подробности и ограничения — [docs/API.md](docs/API.md).

```bash
mkdir -p inputs results
# Положите своё фото в inputs/photo.jpg.
uv venv --python 3.11 .venv-mcp
uv pip install --python .venv-mcp/bin/python -r requirements-mcp.txt
.venv-mcp/bin/python tests/test_smoke.py --url http://127.0.0.1:8080 \
  --image inputs/photo.jpg --output results/smoke
.venv-mcp/bin/python -m unittest discover -s tests -p 'test_mcp_server.py' -v
```

Smoke проверяет статику, реальную модель, хеш и геометрию маски. Дополнительный `--expected-mask inputs/mask.png` сравнивает пиксели с эталоном.

История хранится в браузере, отдельно для адреса/порта. Для переноса скачайте проверку или обращение. Пустая маска не доказывает подлинность; VLM даёт подсказку для осмотра, решение принимает оператор.
