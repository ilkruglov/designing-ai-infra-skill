# Шпаргалка: порядок анализа

Быстрый путь: порядок анализа книги на одной странице и команда калькулятора под каждый вопрос. Подробности — в конспектах глав `references/chapters/`, числа книги — в `references/numbers.md`, типовые ошибки — в `references/fallacies.md`, термины — в `references/glossary.md`.

## Порядок анализа

> «При анализе системы сначала проверьте, помещаются ли данные в память, затем оцените время выполнения по объёму вычислений и операций чтения и записи» — `references/source-book/chapter1.md:504`

| Шаг | Вопрос | Как считать | Что обычно забывают | Книга |
|---|---|---|---|---|
| 0. Задача и требования | Что считается успехом? | задача, требования к качеству и срокам: целевые TTFT, ITL, полное время, доля правильных ответов, бюджет | что throughput без SLO и качества ничего не говорит о пригодности ответа | `references/source-book/preface.md:15`, `references/source-book/chapter3.md:29`, `references/source-book/chapter8.md:616` |
| 1. Память | Помещается ли? | `M_W + M_state + M_work ≤ M_cap` для каждого ускорителя, всё в одних единицах | KV всех одновременных запросов к концу генерации, рабочую область и резерв среды выполнения; scale квантизации; GB против GiB | `references/source-book/chapter1.md:152`, `references/source-book/chapter1.md:189`, `references/source-book/chapter8.md:52` |
| 2. Нижняя граница времени | Быстрее чего нельзя физически? | `T ≥ max(F/Π, R/β)` по пикам нужной точности; назвать результат нижней границей | чтение KV вдобавок к весам; пик другой точности или разреженный пик для плотной задачи | `references/source-book/chapter1.md:263`, `references/source-book/chapter4.md:921` |
| 3. Узкое место | Какой член доминирует и кто кого ждёт? | сравнить `F/Π` и `R/β` (интенсивность против точки пересечения `Π/β`); добавить обмен, перекрытие и очередь с учётом порядка выполнения | обмен между картами при «поровну поделённой» работе; самую загруженную карту вместо средней; задержку последовательных зависимостей | `references/source-book/chapter1.md:401`, `references/source-book/chapter5.md:930`, `references/source-book/chapter6.md:490` |
| 4. Измерение и разрыв | Насколько далеко от предела и почему? | `MFU = F/(Π·T)`, `MBU = R/(β·T)` по измеренному T; разрыв — сначала неполнота модели, затем устранимые накладные расходы | одинаковые определения метрик (включает ли throughput prefill, как считается TPOT); одно изменённое условие на измерение | `references/source-book/chapter1.md:325`, `references/source-book/chapter1.md:357`, `references/source-book/chapter4.md:1003` |

> «Различить две причины можно только измерениями, в каждом из которых изменяется одно условие.» — `references/source-book/chapter1.md:357`

Для нескольких ускорителей тот же порядок идёт по картам: помещаются ли данные на каждую, сколько вычислений и чтения приходится на каждую, какое общее время с учётом зависимостей (`references/source-book/chapter1.md:401`).

## Пять ограничений, которые упускают

Предисловие перечисляет, на чём ошибаются и люди, и модели (`references/source-book/preface.md:15`). Проверить каждое до ответа:

1. Учтено только чтение весов — а чтение KV-кэша механизмом внимания?
2. Время чтения посчитано — а помещаются ли в видеопамять вместе веса, KV-кэш и рабочая область среды выполнения?
3. Скорость взята из пика FLOP/s — успевает ли хранилище непрерывно подавать данные?
4. Работа поделена поровну между ускорителями — где обмен данными между ними?
5. Получена высокая пропускная способность — какова задержка последовательных зависимостей и передач туда и обратно?

> «Пропускную способность и время отклика следует указывать вместе.» — `references/source-book/chapter1.md:438`

## Рычаг под доминирующий член

| Доминирует | Помогает | Не помогает | Конспект |
|---|---|---|---|
| Чтение весов (decode, малый batch) | больше β; меньше байт весов (квантизация); batch до точки перехода `B* = b_W·Π/(2β)` | больше пика FLOP/s | `references/chapters/ch01-ai-infra-basics.md`, `references/chapters/ch08-inference-optimization.md` |
| Чтение KV (длинный контекст, много запросов) | GQA/MLA, окно и сжатие KV, повторное использование префикса | увеличение batch: KV каждого запроса читается отдельно | `references/chapters/ch02-model-architecture.md`, `references/chapters/ch08-inference-optimization.md` |
| Вычисления (prefill, крупный batch) | пик нужной точности, меньше FLOPs; блочный prefill ради ITL соседей | больше пропускной способности HBM | `references/chapters/ch04-accelerators.md`, `references/chapters/ch08-inference-optimization.md` |
| Обмен малыми сообщениями | меньше раундов и задержки запуска α (дерево, слияние коммуникаций) | более широкий канал | `references/chapters/ch06-supernodes.md` |
| Обмен крупными сообщениями | полоса, согласование пар по rail, перекрытие с вычислениями | уменьшение α | `references/chapters/ch06-supernodes.md`, `references/chapters/ch07-datacenter-network.md` |
| Хост и запуск kernel | CUDA Graph, слияние операторов, persistent kernel | более быстрый ускоритель | `references/chapters/ch05-operators-runtime.md` |
| Очередь | загрузка каждого ресурса заметно ниже 1, допуск и отмена | рост средней мощности без учёта всплесков | `references/chapters/ch03-workloads.md`, `references/chapters/ch08-inference-optimization.md` |

## Вопрос → команда → глава

Команды запускаются из каталога скилла; каждая приведённая строка выполнена и завершилась с кодом 0. Конфиги в `scripts/tests/fixtures/configs/` — эталоны тестов на дату книги: для реальной задачи возьмите актуальный `config.json` модели. Устройства — снимок `data/hardware.json` (bojieli/ai-infra-book@d0cc188b, 2026-09-30); любое значение переопределяется `--peak-tflops`, `--bandwidth`, `--memory`. Цены передаются только аргументом. Формат вывода: `--format md` (по умолчанию) или `--format json` перед именем команды.

| Вопрос | Команда | Поле вывода | Глава |
|---|---|---|---|
| Сколько это в GiB, байт на тип, GB/s канала? | `python3 scripts/calc.py units --size "141107412992 B" --to GiB --mbps 400000 --dtype fp8` | `size_GiB`, `link_bytes_per_second`, `dtype_bytes` | 1 — `references/chapters/ch01-ai-infra-basics.md` |
| Параметры, веса, KV на токен, FLOPs prefill и decode? | `python3 scripts/calc.py model --config scripts/tests/fixtures/configs/qwen3-8b.json --context 2048` | `weight_bytes`, `kv_bytes_per_token`, `decode_step_flops` | 2 — `references/chapters/ch02-model-architecture.md` |
| Память, пропускная способность и пик устройства? | `python3 scripts/calc.py device --device h100-sxm` | `memory_bytes`, `bandwidth`, `peak_flops` | 1, 4 — `references/chapters/ch04-accelerators.md` |
| Нижняя граница шага и что доминирует? | `python3 scripts/calc.py roofline --device h100-sxm --flops 140e9 --bytes 70e9` | `step_lower_bound_seconds`, `arithmetic_intensity`, `ridge_point` | 1, 4 — `references/chapters/ch01-ai-infra-basics.md` |
| Сколько запросов поместится, TTFT и TPOT снизу, цена токена? | `python3 scripts/calc.py serving --device rtx-pro6000-blackwell-ws --weights 16381470720 --weight-read 15136811008 --decode-flops 16344154112 --prefill-flops 29688662589440 --kv-per-token 147456 --context 2048 --memory-context 2304 --batch 1 --reserve 2147483648 --price-per-hour 2` | `max_concurrent_requests`, `tpot_lower_bound_seconds`, `ttft_lower_bound_seconds`, `cost_per_million_tokens_lower_bound` | 1, 8 — `references/chapters/ch08-inference-optimization.md` |
| Окупится ли черновик спекулятивного декодирования? | `python3 scripts/calc.py speculative --acceptance 0.75 --draft 4 --plain-step 0.02626` | `expected_tokens_per_round`, `breakeven_round_seconds` | 8 — `references/chapters/ch08-inference-optimization.md` |
| Хватает ли мощности prefill и decode на поток запросов? | `python3 scripts/calc.py queueing --class reason=8192:1025 --rate reason=3.5 --prefill-capacity 38264 --decode-capacity 4664` | `prefill_utilization`, `decode_utilization` | 3, 9 — `references/chapters/ch09-distributed-inference.md` |
| Сколько запросов одновременно в системе (закон Литтла)? | `python3 scripts/calc.py queueing --arrival-rate 4 --time-in-system 0.5` | `in_system` | 3, 8 — `references/chapters/ch03-workloads.md` |
| Сколько стоит кольцевой AllReduce? | `python3 scripts/calc.py ring --devices 8 --message 10240 --bandwidth 450e9 --alpha 0.822e-6` | `ring_allreduce_seconds` | 6, 7 — `references/chapters/ch06-supernodes.md` |
| Кольцо или дерево для этого сообщения? | `python3 scripts/calc.py allreduce --devices 8 --message 10240 --bandwidth 450e9 --alpha 0.822e-6` | `tree_allreduce_seconds`, `ring_tree_crossover_bytes` | 6 — `references/chapters/ch06-supernodes.md` |
| С какого batch чтение KV догоняет чтение весов? | `python3 scripts/calc.py batch-threshold --config scripts/tests/fixtures/configs/qwen3-8b.json --context 8192 --device h100-sxm` | `kv_read_batch_threshold`, `compute_bound_batch_threshold` (здесь «не вычисляется»: шаг Qwen3-8B при 8192 не упирается в вычисления ни при каком batch) | 2, 8 — `references/chapters/ch02-model-architecture.md` |
| FLOPs обучения, состояние ZeRO на GPU, срок? | `python3 scripts/calc.py training --device h100-sxm --config scripts/tests/fixtures/configs/qwen3-8b.json --tokens 8192 --dp 8 --total-tokens 100e9 --devices 8 --mfu 0.4` | `training_flops_per_sequence`, `zero3_state_bytes_per_gpu`, `training_seconds` | 3, 10 — `references/chapters/ch10-training-systems.md` |
| Как часто сохранять checkpoint? | `python3 scripts/calc.py checkpoint --checkpoint-bytes 114670295040 --save-bandwidth 7e9 --devices 1024 --device-mtbf 29122560 --recovery 120` | `first_order_optimal_interval_seconds`; откат всего задания — `--common-job-mtbf` | 10 — `references/chapters/ch10-training-systems.md` |
| Пузырь и утилизация конвейера? | `python3 scripts/calc.py pipeline --stages 4 --microbatches 8 --forward-seconds 0.010 --backward-seconds 0.020` | `pipeline_bubble_ratio`, `pipeline_utilization`, `pipeline_step_seconds` | 6, 10 — `references/chapters/ch10-training-systems.md` |
| Веса, градиенты и оптимизатор на GPU при ZeRO? | `python3 scripts/calc.py training-state --config scripts/tests/fixtures/configs/qwen3-8b.json --dp 8 --stage 3` | `training_state_bytes_per_device` | 3, 10 — `references/chapters/ch10-training-systems.md` |
| Сколько стоит принятая задача с повторами? | `python3 scripts/calc.py cost --input-tokens 10000 --output-tokens 1200 --input-price 2 --output-price 10 --calls 20 --success 0.5` | `cost_per_call`, `cost_per_accepted_task` | 3, 11 — `references/chapters/ch11-resource-scheduling.md` |
| Устройство, периферия или облако: полное время? | `python3 scripts/calc.py edge --upload '30 MB' --download '5 MB' --up-mbps 20 --down-mbps 100 --rtt 0.1 --compute 0.3` | `serial_seconds` | 12 — `references/chapters/ch12-edge-cloud.md` |

Как связаны строки таблицы:

- `model` → `serving`: `weight_bytes` → `--weights`, `decode_weight_read_bytes` → `--weight-read`, `decode_step_flops` → `--decode-flops`, `prefill_flops` → `--prefill-flops`, `kv_bytes_per_token` → `--kv-per-token`. FLOPs шага decode и prefill зависят от длины контекста (при 8192 `decode_step_flops` = 19 968 032 768, при 2048 — 16 344 154 112), поэтому `model --context` и `serving --context` — длина входа: шаг при ней — нижняя граница каждого шага decode. Память — другая длина: `serving --memory-context` = вход + выход − 1, округлённо вверх до блока KV (в примере 2048 + 256 − 1 → 2304). Общий префикс запросов — `serving --shared-prefix-tokens P`: KV префикса хранится один раз, на запрос — `--memory-context` − P.
- `serving --reserve` — резерв среды выполнения и рабочей области; в примере 2 GiB, как рабочая область в разделе 1.2.3 (`references/source-book/chapter1.md:189`). Без него `max_concurrent_requests` считается при нулевом резерве (234 вместо 228). Это верхняя граница: активации, фрагментация и буферы сверх `--reserve` не учтены, а KV считается на `--memory-context`; без этого флага — на `--context`, то есть на длину входа, и ёмкость завышена.
- `queueing --class`: `--prefill-capacity 38264` = 4 × 9566 токенов/s prefill A100, `--decode-capacity 4664` = 4 × 1166 вызовов decode/s H20 — пример 9.2 (`references/source-book/chapter9.md:154`), четыре A100 — пул P, четыре H20 — пул D; пул с загрузкой ≥ 1 не успевает.
- `training --dp` — размер группы шардирования ZeRO, не больше `--devices`: при большем калькулятор отказывает.
- `ring` считает только кольцо, `allreduce` — кольцо и несегментированное дерево `T_tree = 2·log₂n·(α + M/B)`: при 8 картах и 10 KiB `tree_allreduce_seconds` ≈ 5,07 μs против 11,55 μs у кольца, `ring_tree_crossover_bytes` 696 282 B ≈ 680 KiB (8 карт, α = 0,822 μs, 450 GB/s) — ниже дерево быстрее, выше кольцо (`references/source-book/chapter6.md:490`).

Список id устройств снимка:

```bash
python3 -c "import json; print(*[d['id'] for d in json.load(open('data/hardware.json'))['devices']])"
```

Чего калькулятор не делает:

- Незнакомую архитектуру не считает как плотную: `model` печатает «не вычисляется» и перечисляет неподдержанные поля; для гибридной модели число параметров берётся из карточки модели и передаётся `--params`.
- Не симулирует: реальная эффективность ядер, фрагментация, перегрузка сети и хвосты задержек требуют измерения (`references/source-book/chapter1.md:357`).
- Для int8 и других квантизаций `weight_bytes` — нижняя граница без scale и частей в высокой точности; известные накладные расходы добавляет `--quant-overhead-bytes`, в `serving --weights` подставляется реальный размер чекпойнта.

## Ответ

1. Решение первым: помещается или нет, какой член доминирует, какой рычаг.
2. Каждое число — с командой калькулятора или якорем на книгу; нижние границы названы нижними.
3. Числа книги и снимка железа — на дату книги; для закупки и развёртывания сверить с текущей спецификацией производителя и текущими ценами (см. «Drift gate» в `references/source-map.md`).
4. Чего нельзя посчитать — назвать неизвестным и предложить измерение с одним изменённым условием.
