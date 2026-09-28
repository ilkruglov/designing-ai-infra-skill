---
name: designing-ai-infra
description: Use when sizing, designing, reviewing, or debugging AI infrastructure for LLM inference or training — GPU and accelerator memory, KV cache, FLOPs, roofline bounds, batching, TTFT and TPOT, distributed and disaggregated inference, parallelism, supernodes and datacenter networks, checkpointing, resource pools for agents and RL, token and task cost, edge-cloud placement. Not for designing an agent's own logic, prompts, tools or memory (use developing-ai-agents); agent workload on infrastructure belongs here.
---

# Проектирование AI-инфраструктуры

Считай инфраструктуру количественно: сначала память, затем нижняя граница времени, затем узкое место, затем сверка с измерением. Опирайся на русский перевод книги Bojie Li «AI Infra in Depth: Quantitative Analysis and System Design» («AI-инфраструктура изнутри») и на факты текущего проекта. Числа считает `scripts/calc.py`, а не модель в уме.

## Рабочий контракт

1. Сначала собери факты: `config.json` модели, точность весов и KV, ускоритель и его число, классы запросов (вход, выход, общий префикс, интенсивность по окнам), SLO, критерий приёмки ответа, измерения и версии движка. Не называй гипотезу причиной без трассы, лога или замера.
2. Разделяй в ответе:
   - **evidence** — подтверждено конфигом, выводом калькулятора, измерением или текстом книги с якорем;
   - **inference** — вывод из этих данных, в том числе прогноз без измерения на тех же условиях;
   - **unknown** — что не измерено и каким замером это закрыть.
3. Начинай с решения: помещается или нет, какой член доминирует, какой рычаг, при каком пороге решение меняется. Вместо «A всегда лучше B» давай порог.
4. Review и диагностика — read-only: не меняй конфиги, не перезапускай сервисы и не запускай нагрузочные тесты, пока пользователь явно не попросит. Предлагай замер, а не выполняй его на чужой системе.
5. Числа книги верны на дату книги (сентябрь 2026) и в условиях её примеров. `data/hardware.json` — снимок оригинала `bojieli/ai-infra-book@56ecb425` от 2026-09-26. Текущие спецификации ускорителей, цены, тарифы, версии моделей и движков проверяй по первичным источникам и называй дату источника; не можешь проверить — это unknown. Порядок сверки — раздел «Drift gate» в [source-map.md](references/source-map.md).
6. Встроенных конфигов моделей нет. Возьми официальный `config.json` модели с Hugging Face. Незнакомую архитектуру не считай как плотную: калькулятор откажется и перечислит неподдержанные поля, а ты назови это unknown.
7. Единицы — явно: GB (10⁹) или GiB (2³⁰), бит или байт, пик какой точности. Стоимость сравнивай на принятый результат или выполненную задачу, а не во FLOPS на доллар и не в токенах в секунду.
8. По умолчанию укладывай ответ в 1000 слов без учёта вывода калькулятора и заполненного шаблона; из вывода цитируй поля, на которых стоит решение. Длиннее — по явной просьбе.

## Порядок анализа книги

> «При анализе системы сначала проверьте, помещаются ли данные в память, затем оцените время выполнения по объёму вычислений и операций чтения и записи» — `references/source-book/chapter1.md:504`
>
> «а в конце проанализируйте перекрытие операций и ожидание с учётом порядка выполнения» — `references/source-book/chapter1.md:504`

0. **Задача.** Что считается успехом: TTFT, ITL или TPOT (среднее, p95), полное время, доля правильных ответов, срок обучения, бюджет. Без классов запросов и SLO можно ответить только «помещается ли» и «какова нижняя граница».
1. **Память.** Для каждой карты `M_w + M_KV + M_a + M_u ≤ C`: веса, KV всех одновременных запросов к концу генерации, активации и рабочие области, буферы среды выполнения (`references/source-book/chapter8.md:52`).
2. **Нижняя граница времени.** `T ≥ max(F/Π, R/β)` по плотному пику нужной точности; результат называется нижней границей (`references/source-book/chapter1.md:263`).
3. **Узкое место.** Какой член доминирует: интенсивность `F/R` против точки пересечения `Π/β`; затем обмен между картами, очередь, хост и последовательные зависимости. Для нескольких карт — самая загруженная карта, а не средняя: объединение результатов ждёт самую медленную (`references/source-book/chapter9.md:352`, `references/source-book/chapter10.md:600`).
4. **Измерение и разрыв.** `MFU = F/(Π·T)`, `MBU = R/(β·T)` по измеренному T. Разрыв — либо неучтённая в модели работа, либо устранимые накладные расходы; различить их можно только замерами с одним изменённым условием (`references/source-book/chapter1.md:357`).

Пять вопросов любой оценки: что перемещается, в каком объёме, сколько раз, по какому пути и кто ожидает (`references/source-book/chapter1.md:325`).

## Правило калькуляторов

Любое число, от которого зависит решение, считает `scripts/calc.py`; в ответе приводится команда и вывод. Модель собирает входные данные, скрипт считает, модель объясняет. Команды запускаются из каталога скилла; `--format json` удобен для передачи полей между командами.

- `model` читает `config.json` и выдаёт `weight_bytes`, `decode_weight_read_bytes`, `kv_bytes_per_token`, `prefill_flops`, `decode_step_flops`; в `serving` они идут как `--weights`, `--weight-read`, `--kv-per-token`, `--prefill-flops`, `--decode-flops`. Значения всей модели: при TP `serving --tp N --config` сам делит их на карту (KV — по головам KV, сверх их числа KV дублируется), `model --tp N` печатает значения на карту; фиксированное состояние гибридной модели `serving` берёт из `--config` или `--fixed-state-bytes`, окно внимания (`sliding_window`) — из `--config`: KV запроса — последние min(длина, окно) токенов; латентный KV MLA по TP не делится (вывод из 2.3.2 и 6.2.2).
- TTFT, TPOT и память требуют разных длин, и у `serving` их две. **`--context` — длина входа:** с ней же берутся `--prefill-flops` и `--decode-flops`; KV растёт с каждым шагом, поэтому шаг при длине входа — нижняя граница для каждого шага decode (`references/source-book/chapter2.md:266`). **`--memory-context` — длина к концу генерации** для `max_concurrent_requests`: вход + выход − 1 токен, округлённо вверх до блока KV движка (`references/source-book/chapter8.md:52`). Без `--memory-context` обе длины равны `--context`: при длине входа память занижена, а при длине к концу генерации `tpot_lower_bound_seconds` — граница только последнего шага, а не нижняя граница TPOT. В примере ниже вход 4096, выход 512: 4607 → 4608 при блоке 16. Общий префикс запросов задаёт `--shared-prefix-tokens`: его KV хранится в пуле один раз, `--memory-context` остаётся полной длиной запроса (`references/source-book/chapter8.md:268`). Шаг тогда печатается двумя полями: `tpot_lower_bound_seconds` читает префикс один раз на batch (нижняя граница при любом ядре), `tpot_without_prefix_dedup_seconds` — каждым запросом; второе — граница, только если ядро внимания так и читает (допущение, книга этого не описывает).
- Железо берётся из снимка по `--device`; вывод печатает источник и дату. Любое значение переопределяется `--memory`, `--bandwidth`, `--peak-tflops`. Стойку или суперчип из снимка калькулятор без `--allow-aggregate` не примет; у такого агрегата `--peak-tflops` и `--bandwidth` переопределяются только вместе.
- Цены — только аргументом и только текущие (`--price-per-hour`, `--input-price`, `--output-price`, `--cache-read-price`, `--cache-write-price`). В `cost` три категории входа не пересекаются: `--input-tokens` — только некэшированный вход.
- Поля с пометкой «нижняя граница» или «верхняя граница» так и называй в ответе.
- Калькулятор не симулятор: эффективность ядер, фрагментацию, перегрузку сети и хвосты задержек он не моделирует. `ring` считает только кольцо, `allreduce` сравнивает кольцо и дерево; для int8 и int4 `weight_bytes` — нижняя граница без scale (добавь `--quant-overhead-bytes` или подставь размер чекпойнта); для MoE `decode_weight_read_bytes` — эксперты одного токена, а batch читает объединение экспертов: шаг `serving` с ним — нижняя граница при любом batch; `decode_weight_read_bytes_at_batch` из `model --batch` — оценка при равномерной маршрутизации, и шаг с ним — оценка, а не граница; реальное объединение — измерение.

Примеры. Конфиги `scripts/tests/fixtures/configs/` — эталоны тестов на дату книги, а не пресеты: даже при совпадении имени модели скачай текущий `config.json` (`https://huggingface.co/<org>/<model>/resolve/main/config.json` в `.tmp/`). Без сети расчёт по эталонному конфигу — допущение с датой книги, так и назови его. Цены в примерах условные.

| Вопрос | Команда |
|---|---|
| Единицы, байты на тип, скорость канала | `python3 scripts/calc.py units --size '48 GB' --to GiB --mbps 100000 --dtype fp8` |
| Память, полоса, пики устройства и дата снимка | `python3 scripts/calc.py device --device h100-sxm` |
| Веса, KV на токен, FLOPs prefill и decode при длине входа | `python3 scripts/calc.py model --config scripts/tests/fixtures/configs/qwen3-8b.json --context 4096` |
| Нижняя граница шага и доминирующий член | `python3 scripts/calc.py roofline --device h100-sxm --flops 17552113664 --bytes 15740790784` |
| TTFT и TPOT снизу при длине входа, сколько запросов поместится к концу генерации | `python3 scripts/calc.py serving --device h100-sxm --weights 16381470720 --weight-read 15136811008 --decode-flops 17552113664 --prefill-flops 61849981681664 --kv-per-token 147456 --context 4096 --memory-context 4608 --batch 8 --reserve 4294967296` |
| С какого batch чтение KV догоняет чтение весов, а decode упирается в вычисления | `python3 scripts/calc.py batch-threshold --config scripts/tests/fixtures/configs/qwen3-8b.json --context 4096 --device h100-sxm` |
| Загрузка prefill и decode по классам запросов | `python3 scripts/calc.py queueing --class short=4096:512 --class long=12288:256 --rate short=6 --rate long=1.5 --prefill-capacity 90000 --decode-capacity 3600` |
| Сколько запросов в системе (закон Литтла) | `python3 scripts/calc.py queueing --arrival-rate 12 --time-in-system 3` |
| Окупится ли черновик спекулятивного декодирования | `python3 scripts/calc.py speculative --acceptance 0.6 --draft 3 --plain-step 0.006` |
| Время AllReduce: кольцо против дерева (`--bandwidth` — B/s в одном направлении) | `python3 scripts/calc.py allreduce --devices 16 --message 2097152 --bandwidth 50e9 --alpha 1.5e-6` |
| FLOPs обучения, состояние ZeRO на GPU, срок | `python3 scripts/calc.py training --device h100-sxm --config scripts/tests/fixtures/configs/qwen3-8b.json --tokens 2048 --dp 16 --total-tokens 20e9 --devices 16 --mfu 0.38` |
| Период checkpoint; откат всего задания (всплеск потерь) — `--common-job-mtbf` | `python3 scripts/calc.py checkpoint --checkpoint-bytes 2e11 --save-bandwidth 1e10 --devices 768 --device-mtbf 2.5e7 --recovery 300` |
| Пузырь и утилизация конвейера | `python3 scripts/calc.py pipeline --stages 8 --microbatches 32` |
| Веса, градиенты и состояние оптимизатора на GPU по stage ZeRO | `python3 scripts/calc.py training-state --config scripts/tests/fixtures/configs/qwen3-8b.json --dp 16 --stage 3` |
| Стоимость вызова и принятой задачи | `python3 scripts/calc.py cost --input-tokens 5000 --cached-tokens 18000 --output-tokens 700 --input-price 0.8 --cache-read-price 0.08 --output-price 3.2 --calls 12 --success 0.7` |
| Полное время: устройство против облака | `python3 scripts/calc.py edge --upload '4 MB' --download '200 KB' --up-mbps 10 --down-mbps 50 --rtt 0.06 --compute 1.2` |

Полная таблица «вопрос → команда → поле вывода» — в [cheatsheet.md](references/cheatsheet.md); цепочки команд под задачу — в playbooks.

## Ключевые механизмы

### Память и KV

- Бюджет пишется для каждой карты: свободная память одной карты служит вычислениям другой только после смены размещения данных (`references/source-book/chapter8.md:52`).
- Хватает на веса ≠ хватит на целевой параллелизм: KV всех запросов резервируется под состояние к концу генерации, плюс рабочая область и резерв среды (`references/source-book/chapter1.md:438`, `references/source-book/chapter8.md:52`).
- KV на токен задаёт внимание. GQA и MQA сокращают число KV-голов, но `QKᵀ` и `AV` по-прежнему идут по всем головам запросов; MLA хранит на токен узкую латентную переменную, но её объём всё равно растёт с длиной контекста; локальное окно читает только ближайшие позиции; линейное внимание держит состояние фиксированного размера (`references/source-book/chapter2.md:312`, `references/source-book/chapter2.md:334`, `references/source-book/chapter2.md:360`, `references/source-book/chapter2.md:396`).
- Хранится ≠ читается: прирост на токен, хранимое при длине N и читаемое за шаг decode считаются отдельно; у сжатых и индексных схем чтение бывает и меньше, и больше хранимого (`references/source-book/chapter2.md:454`).
- MoE: в каждом слое хранятся все маршрутизируемые эксперты (`references/source-book/chapter2.md:532`); FLOPs растут с числом назначений токенов, а чтение весов — с числом разных экспертов, к которым обратился batch (`references/source-book/chapter2.md:542`).
- Число запросов меняется скачками: остаток памяти должен вместить полное состояние запроса (`references/source-book/chapter2.md:805`). Страничное KV теряет место только в последнем блоке, общий префикс хранится один раз (`references/source-book/chapter8.md:232`).
- Квантизация хранит значения группами со scale; при низкой разрядности метаданные — заметная доля байтов, и она растёт с уменьшением разрядности (`references/source-book/chapter8.md:386`).

### Нижняя граница времени и разрыв с измерением

- `max(F/Π, R/β)` — при полном перекрытии чтения и вычислений; без перекрытия времена складываются (`references/source-book/chapter1.md:263`).
- Decode при малом batch упирается в чтение весов и KV: рост пика FLOP/s его не ускоряет; помогают полоса, меньше байт и повторное использование (`references/source-book/chapter1.md:263`, `references/source-book/chapter1.md:438`).
- Точка перехода `B* = b_W·Π/(2β)` учитывает только веса. С KV каждого запроса `T ≈ max(B·F₁/Π, (R_W + B·R_state)/β)`, и чем длиннее контекст, тем раньше кончается выигрыш batch (`references/source-book/chapter1.md:325`, `references/source-book/chapter2.md:236`).
- Пик зависит от точности входа и накопления; выбор пика может поменять доминирующий член. Выше точки пересечения оценивают MFU, ниже — MBU (`references/source-book/chapter4.md:906`).
- Разрыв с замером: сначала поправить область учёта, затем устранять накладные расходы — нужны оба шага (`references/source-book/chapter4.md:988`).
- Если ускоритель ждёт отправки kernel с хоста, быстрый ускоритель не поможет; CUDA Graph снимает повторяющуюся работу хоста, узлы графа выполняются по-прежнему (`references/source-book/chapter5.md:634`, `references/source-book/chapter5.md:658`). Выигрыш оператора переводи в выигрыш запроса по закону Амдала (`references/source-book/chapter5.md:772`) и критическому пути: ускоренная ветвь может уступить место другой (`references/source-book/chapter5.md:784`).
- Порядок выполнения и форма тайла задают, сколько раз данные проходят через интерфейс памяти (`references/source-book/chapter5.md:182`); большой блок читает меньше, но может получить меньшую пропускную способность (`references/source-book/chapter5.md:225`). Слияние поэлементных операторов убирает запись и чтение промежуточного тензора; ёмкость промежуточных тензоров и объём обращений к ним — разные величины (`references/source-book/chapter5.md:343`).

### Batch, очередь, SLO

- Batch уменьшает чтение весов на выходной токен, но увеличивает объём размещённых данных; KV каждого запроса читается отдельно, чтение KV догоняет чтение весов при `b_KV = ⌈D_w/(L·k)⌉` (`references/source-book/chapter8.md:104`).
- Пропускную способность и задержку указывай вместе: каждый запрос проходит очередь и выполнение всего batch (`references/source-book/chapter1.md:438`).
- Непрерывный batching сокращает общее время, но чужой длинный prefill растягивает ITL текущих запросов; блок prefill — самый крупный, который ещё укладывается в ITL (`references/source-book/chapter8.md:165`, `references/source-book/chapter8.md:191`).
- Нагрузку считай по окнам: время простоя прошлой минуты не спасает от всплеска. Заполненный пул KV держит новые запросы в очереди, даже если часть вычислительных блоков простаивает (`references/source-book/chapter3.md:75`). Закон Литтла `n̄ = λ·T̄` задаёт число запросов и KV в системе (`references/source-book/chapter8.md:598`).
- Выгрузка весов в память хоста освобождает место под KV, но каждый шаг переносит те же байты по каналу: предвыборка перекрывает перенос с вычислениями, не уменьшая число байтов, и если перенос за шаг дольше шага decode, шаг не короче переноса; чистая экономия памяти меньше выгруженного объёма на буфер предвыборки (`references/source-book/chapter8.md:432`).
- Эффективная пропускная способность считает только правильные и вовремя завершённые запросы (`references/source-book/chapter8.md:616`).
- Спекулятивное декодирование окупается, если полный раунд короче `E[N]` обычных шагов, с учётом подготовки черновика (`references/source-book/chapter8.md:536`).

### Распределённый инференс

- Полные реплики, PD, AF и общий KV делят разное; для каждого ресурса нужно `λ·d_r < n_r` (`references/source-book/chapter9.md:17`). При близких потребностях этапов реплики уже сбалансированы, разделение добавит только передачу (`references/source-book/chapter9.md:63`).
- PD в однородном кластере не повышает пропускную способность: GPU-секунды на запрос те же; там обычно сначала применяют блочный prefill. В гетерогенном кластере выигрыш дают различия железа (`references/source-book/chapter9.md:154`).
- Передача KV: её время — минимальное ожидание запроса, а разность скоростей поступления и передачи — темп роста очереди (`references/source-book/chapter9.md:100`).
- Соотношение пулов задаёт состав запросов: попадание префикса смещает ресурсы к D, короткий вывод — к P (`references/source-book/chapter9.md:216`). Если узкое место D, добавление P очередь не сократит (`references/source-book/chapter9.md:793`).
- Большой EP: сначала найди узкое место (горячая группа, dispatch, combine); расширение EP само не увеличивает batch эксперта; отбрасывание токенов и смена top-k меняют модель и балансировкой не считаются (`references/source-book/chapter9.md:352`, `references/source-book/chapter9.md:302`, `references/source-book/chapter9.md:426`).
- Общий KV не возникает из суммы объёмов; маршрутизация с учётом кэша начинается с выбора цели: время целевого запроса или завершение всех задач (`references/source-book/chapter9.md:497`, `references/source-book/chapter9.md:609`). Вывод, возвращённый пользователю, не означает сохранённого KV (`references/source-book/chapter9.md:716`).

### Обучение и checkpoint

- `6ND` — грубая оценка; поэлементный расчёт добавляет внимание и словарную голову, и эффективность оценивают по скорректированной границе (`references/source-book/chapter3.md:442`). Срок — `F/(p·Π·MFU)` плюс ожидания и сбои (`references/source-book/chapter10.md:116`).
- Adam в смешанной точности — около 16 байт на обучаемый параметр (с градиентами FP32 — 18, формат указывай явно); копия операндов в FP8 не сводит его к 1 байту; в MoE состояние считается по всем экспертам (`references/source-book/chapter10.md:59`, `references/source-book/chapter3.md:384`).
- ZeRO/FSDP шардирует постоянное состояние, но постоянное состояние и пик памяти уменьшаются в разной степени: размер буфера полных весов задаёт модуль, а не число GPU, и активации живут рядом с ним (`references/source-book/chapter10.md:151`).
- Повторное вычисление активаций обменивает вычисления на память; выгрузка на хост превращает нехватку памяти в вопрос, успеет ли передача: ожидание `max(0, V/B − W)`, где W — окно между возможным началом передачи и моментом, когда данные нужны (`references/source-book/chapter10.md:205`, `references/source-book/chapter10.md:237`).
- Если batch уже значительно больше критического, добавленные GPU сокращают лишь время шага (сильное масштабирование), а не число шагов (`references/source-book/chapter10.md:21`).
- Конвейер: `u = m/(m + p − 1)`; 1F1B не уменьшает пузырь, а раньше освобождает активации; ни одно расписание не выигрывает сразу по времени и по резидентности (`references/source-book/chapter10.md:322`).
- Checkpoint: `τ* = √(2c/λ)`; по статистике Meta частота прерываний задания пропорциональна числу ускорителей; минимум пологий (`references/source-book/chapter10.md:553`); асинхронный снимок годен для восстановления только после фиксации (`references/source-book/chapter10.md:527`). Шаг задаёт самый медленный узел, а не среднее (`references/source-book/chapter10.md:600`).
- RL: считай пригодные для обучения траектории или токены в секунду; при переключении этапов два набора данных могут не поместиться вместе (`references/source-book/chapter10.md:629`, `references/source-book/chapter10.md:645`).

### Сети и коллективные операции

- Кольцо `2(n−1)α + 2(n−1)M/(nB)`, дерево `2·log₂n·(α + M/B)`: малые сообщения выигрывают от меньшего числа раундов (дерево), крупные — у кольца; при малом M удвоение полосы почти ничего не даёт (`references/source-book/chapter6.md:473`).
- Граница `M = α·B`: выше неё повышай полосу, ниже — сокращай число передач (`references/source-book/chapter7.md:208`).
- Выбор параллелизма — по тому, что и как часто передаётся. DP объединяет градиенты всех участников редукцией; её алгоритм и размещение участников по серверам задают, какая часть объёма идёт между серверами (`references/source-book/chapter7.md:120`). При TP карта зачастую получает лишь часть вклада в выход, частичные суммы складываются внутри слоя, и этот обмен прямо задерживает следующие операторы; PP переносит обмен на границы стадий, но загрузка конвейера зависит от числа готовых micro-batch; EP — dispatch и combine, объём зависит от размещения экспертов (`references/source-book/chapter6.md:80`, `references/source-book/chapter7.md:208`, `references/source-book/chapter6.md:283`, `references/source-book/chapter2.md:573`).
- TP шире числа KV-голов дублирует KV (`references/source-book/chapter6.md:131`); для decode с малым параллелизмом важнее сократить синхронизацию между серверами внутри слоя и ускорить запуск и завершение операций (`references/source-book/chapter7.md:908`).
- Сквозная полоса — самый медленный участок, полосы участков не складываются (`references/source-book/chapter7.md:41`). Иерархическая редукция через несколько независимых сетевых карт почти ничего не стоит и даёт каждой карте свою часть; при общем межсерверном интерфейсе она быстрее, только если дополнительная локальная редукция дешевле сэкономленной удалённой передачи (`references/source-book/chapter7.md:120`); сетевая карта работает, только если алгоритм направляет на неё байты (`references/source-book/chapter7.md:269`).
- Перекрытие обмена и вычислений измеряй при одновременном запуске, а не вычитанием; фиксированный All-to-All не равен реальному обмену MoE (`references/source-book/chapter6.md:540`).
- Больший суперузел даёт ёмкость и суммарную полосу, но не скорость одного сеанса (`references/source-book/chapter6.md:1004`). Управление перегрузкой должно не только останавливать рост очереди, но и оставлять запас полосы на её опустошение (`references/source-book/chapter7.md:700`).

### Пул ресурсов, нагрузка агентов и стоимость

- Потребность ресурса = интенсивность задач × потребность на задачу; в сквозном примере книги первой кончается память сред, а не CPU, и медленная модель увеличивает эту память, потому что среды живут дольше (`references/source-book/chapter11.md:62`).
- Остановка сама по себе не освобождает память: пока среду и KV не выгрузили и не пересоздали, они занимают ёмкость, поэтому ожидание инструмента — вопрос ёмкости (`references/source-book/chapter11.md:33`, `references/source-book/chapter3.md:29`, `references/source-book/chapter9.md:76`). Держать, приостановить или пересоздать среду — решается по окну ожидания и цене сохранения и восстановления (`references/source-book/chapter11.md:203`).
- Рост префикса: кэш сокращает пересчёт, но внимание по-прежнему читает весь старый контекст; выгода зависит от места правки (`references/source-book/chapter3.md:151`), а динамическая метка в начале промпта ломает повторное использование (`references/source-book/chapter8.md:310`).
- Параллельные ветви сокращают критический путь, но умножают состояние в памяти (`references/source-book/chapter3.md:209`).
- Суммы свободных GPU недостаточно: групповое выделение требует ресурсов на подходящих узлах одновременно (`references/source-book/chapter11.md:294`).
- RL: итерация = генерация + проверка + обновление + публикация весов; добавление ускорителей rollout сокращает только параллелизуемую часть генерации (`references/source-book/chapter11.md:370`).
- Стоимость успешной задачи `c/p`; новый метод выгоден при `p₂/p₁ > c₂/c₁` (`references/source-book/chapter3.md:173`); неуспешные попытки остаются в числителе (`references/source-book/chapter11.md:625`). При равной вероятности успеха и достаточной производительности свой сервис выгоднее API при `N > F/(c − v)` (`references/source-book/chapter11.md:552`).

### Устройство, периферия и облако

- `T = S_u/B_u + R + T_c + S_d/B_d`; уменьшение файла сокращает только отправку (`references/source-book/chapter12.md:13`). Облако оправдано, лишь если сэкономленное время вычислений больше добавленных передачи и ожидания (`references/source-book/chapter12.md:3`).
- Устройство: полоса памяти ограничивает шаг decode, ёмкость — модель и контекст, энергия — длительность работы. Граница выше срока исключает вариант, граница ниже срока лишь не исключает его (`references/source-book/chapter12.md:156`).
- Мультимодальный запрос — цепочка E → P → D: размер изображения задаёт работу кодировщика, число визуальных токенов — работу языковой части; считай их отдельно (`references/source-book/chapter3.md:286`). В реальном времени важен запас буфера воспроизведения: предварительная буферизация увеличивает запас, но откладывает начало (`references/source-book/chapter3.md:334`).
- В Computer Use ожидание каждого раунда накапливается за задачу (`references/source-book/chapter12.md:114`).
- Установление соединения платится один раз, передача — в каждом раунде; окно и ACK задают эффективную полосу (`references/source-book/chapter12.md:319`, `references/source-book/chapter12.md:329`).
- Перенос сеанса оправдан, если ожидаемая экономия покрывает подготовку к переносу с запасом (`references/source-book/chapter12.md:259`, `references/source-book/chapter12.md:630`).

## Маршрут по задаче

| Запрос выглядит как | Playbook | Шаблон | Конспект |
|---|---|---|---|
| сколько карт, поместится ли, TTFT и TPOT, SLO, цена миллиона токенов | [size-inference](references/playbooks/size-inference.md) | [sizing-sheet](references/templates/sizing-sheet.md) | [ch08](references/chapters/ch08-inference-optimization.md), [ch01](references/chapters/ch01-ai-infra-basics.md) |
| оптимизация одного экземпляра: спекулятивное декодирование, квантование, сжатие и выгрузка KV, кэш префикса в обычном чате | [size-inference](references/playbooks/size-inference.md) | [sizing-sheet](references/templates/sizing-sheet.md) | [ch08](references/chapters/ch08-inference-optimization.md) (8.3–8.5) |
| какая модель или архитектура влезет: GQA, MLA, MoE, окно, точность, контекст | [compare-model-architectures](references/playbooks/compare-model-architectures.md) | [sizing-sheet](references/templates/sizing-sheet.md) | [ch02](references/chapters/ch02-model-architecture.md) |
| замер хуже расчёта, throughput не растёт, TTFT растёт под нагрузкой | [diagnose-serving](references/playbooks/diagnose-serving.md) | [bottleneck-diagnosis](references/templates/bottleneck-diagnosis.md) | [ch08](references/chapters/ch08-inference-optimization.md), [ch05](references/chapters/ch05-operators-runtime.md) |
| медленный kernel, слияние, тайл, CUDA Graph, хост, динамические формы | [optimize-operators-runtime](references/playbooks/optimize-operators-runtime.md) | [bottleneck-diagnosis](references/templates/bottleneck-diagnosis.md) | [ch05](references/chapters/ch05-operators-runtime.md) |
| PD или AF, пулы P и D, размещение экспертов, общий KV, маршрутизация, отказы | [design-distributed-inference](references/playbooks/design-distributed-inference.md) | [deployment-decision](references/templates/deployment-decision.md) | [ch09](references/chapters/ch09-distributed-inference.md) |
| выбор ускорителя, суперузла, TP, топологии и сети | [choose-hardware-and-cluster](references/playbooks/choose-hardware-and-cluster.md) | [deployment-decision](references/templates/deployment-decision.md) | [ch04](references/chapters/ch04-accelerators.md), [ch06](references/chapters/ch06-supernodes.md), [ch07](references/chapters/ch07-datacenter-network.md) |
| обучение: карты и срок, ZeRO, конвейер, checkpoint, сбои; в RL — алгоритм, синхронизация весов, память этапов | [plan-training](references/playbooks/plan-training.md) | [training-plan](references/templates/training-plan.md) | [ch10](references/chapters/ch10-training-systems.md), [ch03](references/chapters/ch03-workloads.md) |
| пул для агентов и RL: соотношение rollout и обучения, песочницы; свой сервис или API, стоимость задачи | [plan-resource-pool](references/playbooks/plan-resource-pool.md) | [resource-pool-plan](references/templates/resource-pool-plan.md) | [ch11](references/chapters/ch11-resource-scheduling.md) |
| нагрузка агентов на инфраструктуру: рост префикса, ожидание инструментов, среды RL | [plan-resource-pool](references/playbooks/plan-resource-pool.md), кэш префикса — [size-inference](references/playbooks/size-inference.md) | [resource-pool-plan](references/templates/resource-pool-plan.md) | [ch03](references/chapters/ch03-workloads.md) (3.2), [ch11](references/chapters/ch11-resource-scheduling.md), [ch08](references/chapters/ch08-inference-optimization.md) (8.3.3), [ch09](references/chapters/ch09-distributed-inference.md) (разделы книги 9.1.3 и 9.5: время пребывания состояния, распределённый KV) |
| устройство, периферия или облако; сеть, миграция сеанса | [design-edge-cloud](references/playbooks/design-edge-cloud.md) | [deployment-decision](references/templates/deployment-decision.md) | [ch12](references/chapters/ch12-edge-cloud.md) |
| симптом известен, причина нет | [fallacies.md](references/fallacies.md) | — | playbook по области |
| нужно число книги или что говорит книга по теме | [numbers.md](references/numbers.md), [source-map.md](references/source-map.md) | — | нужный конспект |

Логика, промпты, инструменты, контекст и память самого агента — это скилл `developing-ai-agents`, не этот. Сюда относится только нагрузка агента на инфраструктуру: сколько KV, сред, CPU, памяти и денег она требует.

Загружай только нужные файлы. Не помещай всю книгу в контекст.

## Формат результата

1. **Решение** — вердикт и порог, при котором он меняется; что не делать сейчас.
2. **Расчёт** — команды `scripts/calc.py` и их вывод; нижние и верхние границы названы так.
3. **Evidence / inference / unknown** — отдельно числа книги с датой, данные снимка железа с датой и то, что требует проверки по текущей спецификации или прайсу.
4. **Узкое место и рычаг** — доминирующий член и что его сдвигает; что не поможет.
5. **Что измерить** — замер с одним изменённым условием и метрика, которая подтвердит или опровергнет вывод.
6. **Риски** — хвосты задержек, всплески нагрузки, сбои, дрейф цен и железа.
7. **Источники книги** — `references/source-book/*.md:line`, только для реально применённых механизмов.

Для развёрнутого артефакта заполни шаблон из маршрута: у каждого есть поля «Допущения», «Что посчитал калькулятор» и «Что требует измерения».

## Сноски книги

Сноски книги указывают на `../calculations/`, `../experiments/`, `../references/` и `../research/` (а также `../case-studies/`, `../archive/`). Это пути в репозитории оригинала `https://github.com/bojieli/ai-infra-book` на коммите `56ecb425b07ea6d16e891cba87bf7db416927d09`, а не локальные файлы скилла; `../references/` оригинала — не `references/` скилла. Как разрешать такие пути — раздел «Ссылки сносок книги» в [source-map.md](references/source-map.md).

## Все материалы

**Быстрый путь и справочники:** [cheatsheet.md](references/cheatsheet.md) — порядок анализа на одной странице и команда под каждый вопрос · [numbers.md](references/numbers.md) — числа книги с условиями и якорями · [fallacies.md](references/fallacies.md) — заблуждения по симптомам: почему неверно и как проверить · [glossary.md](references/glossary.md) — термины · [source-map.md](references/source-map.md) — карта книги по темам, drift gate и сноски · `references/source-map.lock.json` — sha256 строк-якорей, по нему валидатор ловит сдвиг текста книги.

**Конспекты глав:** [ch00 предисловие и метод](references/chapters/ch00-preface.md) · [ch01 основы, единицы, нижняя граница](references/chapters/ch01-ai-infra-basics.md) · [ch02 архитектура модели, KV, MoE](references/chapters/ch02-model-architecture.md) · [ch03 нагрузки инференса, агентов и обучения](references/chapters/ch03-workloads.md) · [ch04 ускорители](references/chapters/ch04-accelerators.md) · [ch05 операторы и среда выполнения](references/chapters/ch05-operators-runtime.md) · [ch06 суперузлы и параллелизм](references/chapters/ch06-supernodes.md) · [ch07 сеть дата-центра](references/chapters/ch07-datacenter-network.md) · [ch08 оптимизация инференса](references/chapters/ch08-inference-optimization.md) · [ch09 распределённый инференс](references/chapters/ch09-distributed-inference.md) · [ch10 системы обучения](references/chapters/ch10-training-systems.md) · [ch11 пул ресурсов и среды](references/chapters/ch11-resource-scheduling.md) · [ch12 устройство, периферия и облако](references/chapters/ch12-edge-cloud.md)

**Процедуры:** [size-inference](references/playbooks/size-inference.md) · [diagnose-serving](references/playbooks/diagnose-serving.md) · [design-distributed-inference](references/playbooks/design-distributed-inference.md) · [plan-training](references/playbooks/plan-training.md) · [choose-hardware-and-cluster](references/playbooks/choose-hardware-and-cluster.md) · [optimize-operators-runtime](references/playbooks/optimize-operators-runtime.md) · [plan-resource-pool](references/playbooks/plan-resource-pool.md) · [design-edge-cloud](references/playbooks/design-edge-cloud.md) · [compare-model-architectures](references/playbooks/compare-model-architectures.md)

**Артефакты:** [sizing-sheet](references/templates/sizing-sheet.md) · [bottleneck-diagnosis](references/templates/bottleneck-diagnosis.md) · [deployment-decision](references/templates/deployment-decision.md) · [training-plan](references/templates/training-plan.md) · [resource-pool-plan](references/templates/resource-pool-plan.md)

**Полный текст книги:** `references/source-book/` — `preface.md` и `chapter1.md`…`chapter12.md`, побайтно из `book-ru/book` перевода на пине `c791c07c8370155474d84b635d41d142f54f4fc9`; рисунков нет, подписи и таблицы остались. К главе обращайся за дословной формулировкой и условиями примера.

**Калькулятор и данные:** `scripts/calc.py` — CLI над пакетом `scripts/infra_calc/` (команды `model`, `roofline`, `batch-threshold`, `serving`, `training`, `training-state`, `pipeline`, `checkpoint`, `speculative`, `ring`, `allreduce`, `cost`, `edge`, `device`, `units`, `queueing`; справка — `python3 scripts/calc.py --help`) · `data/hardware.json` — снимок ускорителей оригинала: 151 запись, у каждого числа источник и локатор.
