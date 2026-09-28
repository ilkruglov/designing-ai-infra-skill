# Карта источников

Канонический источник скилла — русский перевод книги Bojie Li «AI Infra in Depth: Quantitative Analysis and System Design» («AI-инфраструктура изнутри»), лежащий в `references/source-book/` целиком: предисловие и 12 глав, побайтно из перевода. Рисунки не включены, подписи и таблицы остались в тексте.

| Источник | Репозиторий | Пин |
|---|---|---|
| Перевод | `https://github.com/ilkruglov/ai-infra-book`, ветка `russian-community-edition`, каталог `book-ru/book` | `c791c07c8370155474d84b635d41d142f54f4fc9` |
| Оригинал и калькуляторы автора | `https://github.com/bojieli/ai-infra-book` | `56ecb425b07ea6d16e891cba87bf7db416927d09` |
| Снимок ускорителей `data/hardware.json` | `calculations/configs/hardware.json` оригинала на том же пине | sha256 `9c4639ec65ca3b808923a7f332114f800c7930635b6fdac08f49686cddbf0ba2` |

Пины, sha256 снимка и каждого файла книги записаны в `SOURCE.json`; валидатор их сверяет. Лицензия всех источников — Apache-2.0.

Ссылка вида `references/source-book/chapterN.md:LINE` указывает на строку-заголовок раздела. Все якоря зафиксированы в `references/source-map.lock.json` вместе с sha256 строки: если текст книги сдвинется, расхождение найдёт валидатор, а не читатель.

Конспект главы обычно отвечает быстрее исходного текста: в нём механизмы, таблицы решений, заблуждения и команды калькулятора. К главе обращайтесь за дословной формулировкой, полным выводом или условиями примера.

## Темы

| Тема | Первичный раздел | Конспект |
|---|---|---|
| Метод книги: оценка порядка величин и вывод решений из ограничений | `references/source-book/preface.md:15`, `references/source-book/preface.md:47` | `references/chapters/ch00-preface.md` |
| Структура книги и маршруты чтения | `references/source-book/preface.md:63`, `references/source-book/preface.md:96` | `references/chapters/ch00-preface.md` |
| Шесть уровней AI-системы | `references/source-book/chapter1.md:31` | `references/chapters/ch01-ai-infra-basics.md` |
| Путь одного запроса, суперузел, scale-up и scale-out | `references/source-book/chapter1.md:57` | `references/chapters/ch01-ai-infra-basics.md` |
| Порядки величин и закон Амдала | `references/source-book/chapter1.md:101` | `references/chapters/ch01-ai-infra-basics.md` |
| Ёмкость, FLOP/FLOPs/FLOP/s, MFU и MBU | `references/source-book/chapter1.md:152` | `references/chapters/ch01-ai-infra-basics.md` |
| Единицы: GB и GiB, биты и байты, проверка ёмкости | `references/source-book/chapter1.md:189` | `references/chapters/ch01-ai-infra-basics.md` |
| Нижняя граница шага `max(F/Π, R/β)` и точка перехода B* | `references/source-book/chapter1.md:229`, `references/source-book/chapter1.md:263` | `references/chapters/ch01-ai-infra-basics.md` |
| Сверка с измерением и разбор разрыва | `references/source-book/chapter1.md:325`, `references/source-book/chapter1.md:357` | `references/chapters/ch01-ai-infra-basics.md` |
| Требования определяют архитектуру: TPU, SmartNIC, Unified Bus | `references/source-book/chapter1.md:361`, `references/source-book/chapter1.md:401`, `references/source-book/chapter1.md:417` | `references/chapters/ch01-ai-infra-basics.md` |
| Конфигурация модели и объём вычислений прямого прохода | `references/source-book/chapter2.md:45`, `references/source-book/chapter2.md:146` | `references/chapters/ch02-model-architecture.md` |
| Prefill, decode, KV на токен, хранение против доступа | `references/source-book/chapter2.md:220`, `references/source-book/chapter2.md:236`, `references/source-book/chapter2.md:266` | `references/chapters/ch02-model-architecture.md` |
| MHA, GQA, MQA и MLA | `references/source-book/chapter2.md:312`, `references/source-book/chapter2.md:334` | `references/chapters/ch02-model-architecture.md` |
| Окно, сжатие, линейное и гибридное внимание | `references/source-book/chapter2.md:360`, `references/source-book/chapter2.md:396`, `references/source-book/chapter2.md:418` | `references/chapters/ch02-model-architecture.md` |
| Сколько хранится и читается на токен: сравнение моделей, CED | `references/source-book/chapter2.md:454` | `references/chapters/ch02-model-architecture.md` |
| MoE: маршрутизация, чтение объединения экспертов, общие эксперты | `references/source-book/chapter2.md:532`, `references/source-book/chapter2.md:542`, `references/source-book/chapter2.md:573` | `references/chapters/ch02-model-architecture.md` |
| Форма сети и MTP | `references/source-book/chapter2.md:593`, `references/source-book/chapter2.md:621` | `references/chapters/ch02-model-architecture.md` |
| Ресурсы всей модели, точность и ёмкость | `references/source-book/chapter2.md:643`, `references/source-book/chapter2.md:805`, `references/source-book/chapter2.md:861` | `references/chapters/ch02-model-architecture.md` |
| Prefill и decode в потоке; метрики нагрузки: TTFT, p95, площадь состояния | `references/source-book/chapter3.md:13`, `references/source-book/chapter3.md:29`, `references/source-book/chapter3.md:75` | `references/chapters/ch03-workloads.md` |
| Многораундовый диалог, агенты, ожидание инструментов | `references/source-book/chapter3.md:151`, `references/source-book/chapter3.md:201`, `references/source-book/chapter3.md:209` | `references/chapters/ch03-workloads.md` |
| Мультимодальность и взаимодействие в реальном времени | `references/source-book/chapter3.md:286`, `references/source-book/chapter3.md:324`, `references/source-book/chapter3.md:334` | `references/chapters/ch03-workloads.md` |
| Вычисления обучения: 6ND против поэлементного расчёта | `references/source-book/chapter3.md:384`, `references/source-book/chapter3.md:442` | `references/chapters/ch03-workloads.md` |
| Нагрузка RL: rollout, проверка, синхронизация весов | `references/source-book/chapter3.md:476`, `references/source-book/chapter3.md:486`, `references/source-book/chapter3.md:506` | `references/chapters/ch03-workloads.md` |
| Scaling Law, бюджеты обучения, стоимость жизненного цикла | `references/source-book/chapter3.md:553`, `references/source-book/chapter3.md:611`, `references/source-book/chapter3.md:667` | `references/chapters/ch03-workloads.md` |
| Состав ускорителя, площадь, энергия, корпус | `references/source-book/chapter4.md:65`, `references/source-book/chapter4.md:89` | `references/chapters/ch04-accelerators.md` |
| Матричные и векторные блоки, низкая точность | `references/source-book/chapter4.md:214`, `references/source-book/chapter4.md:250`, `references/source-book/chapter4.md:314` | `references/chapters/ch04-accelerators.md` |
| Иерархия памяти: HBM, унифицированная память, кэши | `references/source-book/chapter4.md:356`, `references/source-book/chapter4.md:396`, `references/source-book/chapter4.md:420` | `references/chapters/ch04-accelerators.md` |
| Перемещение данных и межсоединения внутри и вне корпуса | `references/source-book/chapter4.md:500`, `references/source-book/chapter4.md:645`, `references/source-book/chapter4.md:666` | `references/chapters/ch04-accelerators.md` |
| Эволюция NVIDIA, Ascend, Apple | `references/source-book/chapter4.md:706`, `references/source-book/chapter4.md:760`, `references/source-book/chapter4.md:798` | `references/chapters/ch04-accelerators.md` |
| Специализированные архитектуры: TPU, Groq, Graphcore, Cerebras, фиксированные веса | `references/source-book/chapter4.md:828`, `references/source-book/chapter4.md:840`, `references/source-book/chapter4.md:854` | `references/chapters/ch04-accelerators.md` |
| Roofline и выбор ускорителя на одной модели | `references/source-book/chapter4.md:906`, `references/source-book/chapter4.md:959` | `references/chapters/ch04-accelerators.md` |
| Прогноз против измерения; ёмкость, скорость, энергия, стоимость | `references/source-book/chapter4.md:988`, `references/source-book/chapter4.md:1041` | `references/chapters/ch04-accelerators.md` |
| Kernel launch, копирование, stream и event | `references/source-book/chapter5.md:25`, `references/source-book/chapter5.md:37`, `references/source-book/chapter5.md:69` | `references/chapters/ch05-operators-runtime.md` |
| Время отправки и готовности; выполнение kernel на SM | `references/source-book/chapter5.md:96`, `references/source-book/chapter5.md:112` | `references/chapters/ch05-operators-runtime.md` |
| Тайлинг, повторное чтение, bank shared memory | `references/source-book/chapter5.md:182`, `references/source-book/chapter5.md:225`, `references/source-book/chapter5.md:283` | `references/chapters/ch05-operators-runtime.md` |
| Слияние операторов, двойная буферизация, FlashAttention | `references/source-book/chapter5.md:343`, `references/source-book/chapter5.md:369`, `references/source-book/chapter5.md:401` | `references/chapters/ch05-operators-runtime.md` |
| Компилятор: преобразования циклов, AKG, выбор по измерению | `references/source-book/chapter5.md:491`, `references/source-book/chapter5.md:568`, `references/source-book/chapter5.md:594` | `references/chapters/ch05-operators-runtime.md` |
| CUDA Graph, динамические формы, persistent kernel | `references/source-book/chapter5.md:658`, `references/source-book/chapter5.md:710`, `references/source-book/chapter5.md:742` | `references/chapters/ch05-operators-runtime.md` |
| От локального выигрыша к запросу: Амдал и критический путь | `references/source-book/chapter5.md:772`, `references/source-book/chapter5.md:784`, `references/source-book/chapter5.md:812` | `references/chapters/ch05-operators-runtime.md` |
| Экземпляр и суперузел; сквозной пример Qwen3-235B-A22B | `references/source-book/chapter6.md:13`, `references/source-book/chapter6.md:25`, `references/source-book/chapter6.md:70` | `references/chapters/ch06-supernodes.md` |
| Шесть видов параллелизма; DP и TP | `references/source-book/chapter6.md:80`, `references/source-book/chapter6.md:117`, `references/source-book/chapter6.md:131` | `references/chapters/ch06-supernodes.md` |
| SP и CP | `references/source-book/chapter6.md:243`, `references/source-book/chapter6.md:255` | `references/chapters/ch06-supernodes.md` |
| PP и EP | `references/source-book/chapter6.md:283`, `references/source-book/chapter6.md:312` | `references/chapters/ch06-supernodes.md` |
| Комбинирование видов параллелизма и масштаб модели | `references/source-book/chapter6.md:352`, `references/source-book/chapter6.md:391`, `references/source-book/chapter6.md:429` | `references/chapters/ch06-supernodes.md` |
| Коллективные коммуникации: ring, tree, конкуренция с вычислениями | `references/source-book/chapter6.md:454`, `references/source-book/chapter6.md:473`, `references/source-book/chapter6.md:540` | `references/chapters/ch06-supernodes.md` |
| Физическая организация суперузлов: топологии, NVLink, TPU, Unified Bus | `references/source-book/chapter6.md:599`, `references/source-book/chapter6.md:705`, `references/source-book/chapter6.md:735` | `references/chapters/ch06-supernodes.md` |
| Пул памяти и общая ёмкость под KV | `references/source-book/chapter6.md:804`, `references/source-book/chapter6.md:822`, `references/source-book/chapter6.md:847` | `references/chapters/ch06-supernodes.md` |
| Размер суперузла и выбор стратегии разбиения | `references/source-book/chapter6.md:873`, `references/source-book/chapter6.md:962`, `references/source-book/chapter6.md:1004` | `references/chapters/ch06-supernodes.md` |
| Модель трафика, rail, сеть Clos, переподписка | `references/source-book/chapter7.md:17`, `references/source-book/chapter7.md:41` | `references/chapters/ch07-datacenter-network.md` |
| DP между серверами и иерархические кольца | `references/source-book/chapter7.md:120` | `references/chapters/ch07-datacenter-network.md` |
| TP, PP и EP между серверами | `references/source-book/chapter7.md:208`, `references/source-book/chapter7.md:239` | `references/chapters/ch07-datacenter-network.md` |
| Несколько сетевых карт и многорельсовая топология | `references/source-book/chapter7.md:269`, `references/source-book/chapter7.md:299` | `references/chapters/ch07-datacenter-network.md` |
| Удалённый доступ: запись и уведомление, Load/Store против Read/Write, слоты в полёте | `references/source-book/chapter7.md:319`, `references/source-book/chapter7.md:367`, `references/source-book/chapter7.md:457` | `references/chapters/ch07-datacenter-network.md` |
| Буферы, Jetty, порядок, обратное давление | `references/source-book/chapter7.md:524`, `references/source-book/chapter7.md:544`, `references/source-book/chapter7.md:640` | `references/chapters/ch07-datacenter-network.md` |
| Перегрузка: incast, PFC, ECMP, взаимная блокировка | `references/source-book/chapter7.md:700`, `references/source-book/chapter7.md:758`, `references/source-book/chapter7.md:812` | `references/chapters/ch07-datacenter-network.md` |
| Критический путь обмена и иерархическая сеть | `references/source-book/chapter7.md:834`, `references/source-book/chapter7.md:856`, `references/source-book/chapter7.md:908` | `references/chapters/ch07-datacenter-network.md` |
| 1024 ускорителя при растущем суперузле; заблуждения развёртывания сети | `references/source-book/chapter7.md:938`, `references/source-book/chapter7.md:1009` | `references/chapters/ch07-datacenter-network.md` |
| Жизненный цикл запроса и бюджет памяти сервинга | `references/source-book/chapter8.md:24`, `references/source-book/chapter8.md:52` | `references/chapters/ch08-inference-optimization.md` |
| TTFT, ITL и SLO | `references/source-book/chapter8.md:82` | `references/chapters/ch08-inference-optimization.md` |
| Batching: повторное использование весов, непрерывная пакетная обработка | `references/source-book/chapter8.md:104`, `references/source-book/chapter8.md:165` | `references/chapters/ch08-inference-optimization.md` |
| Блочный prefill, бюджет токенов, формы CUDA Graph | `references/source-book/chapter8.md:191`, `references/source-book/chapter8.md:218` | `references/chapters/ch08-inference-optimization.md` |
| Paged KV, копирование при записи, освобождение | `references/source-book/chapter8.md:232`, `references/source-book/chapter8.md:268` | `references/chapters/ch08-inference-optimization.md` |
| Кэш префикса, допуск в кэш, вытеснение, выгрузка и повторное вычисление | `references/source-book/chapter8.md:310`, `references/source-book/chapter8.md:354` | `references/chapters/ch08-inference-optimization.md` |
| Квантование весов, сжатие KV, выгрузка | `references/source-book/chapter8.md:386`, `references/source-book/chapter8.md:398`, `references/source-book/chapter8.md:432` | `references/chapters/ch08-inference-optimization.md` |
| Спекулятивное декодирование | `references/source-book/chapter8.md:504`, `references/source-book/chapter8.md:536`, `references/source-book/chapter8.md:572` | `references/chapters/ch08-inference-optimization.md` |
| Поток запросов, эффективная пропускная способность, стоимость конфигурации | `references/source-book/chapter8.md:598`, `references/source-book/chapter8.md:616`, `references/source-book/chapter8.md:642` | `references/chapters/ch08-inference-optimization.md` |
| Реплики, PD, AF, общий KV; время пребывания состояния | `references/source-book/chapter9.md:17`, `references/source-book/chapter9.md:63`, `references/source-book/chapter9.md:76` | `references/chapters/ch09-distributed-inference.md` |
| PD-разделение: передача KV и соотношение пулов | `references/source-book/chapter9.md:90`, `references/source-book/chapter9.md:100`, `references/source-book/chapter9.md:154` | `references/chapters/ch09-distributed-inference.md` |
| AF-разделение: CPU и GPU, межсерверное AF | `references/source-book/chapter9.md:240`, `references/source-book/chapter9.md:284`, `references/source-book/chapter9.md:302` | `references/chapters/ch09-distributed-inference.md` |
| Перекос при большом EP и реплики экспертов | `references/source-book/chapter9.md:352`, `references/source-book/chapter9.md:426`, `references/source-book/chapter9.md:455` | `references/chapters/ch09-distributed-inference.md` |
| Распределённый KV: многоуровневое хранилище, персистентность, маршрутизация с учётом кэша | `references/source-book/chapter9.md:509`, `references/source-book/chapter9.md:595`, `references/source-book/chapter9.md:609` | `references/chapters/ch09-distributed-inference.md` |
| Запуск, реконфигурация, частичные сбои | `references/source-book/chapter9.md:674`, `references/source-book/chapter9.md:696`, `references/source-book/chapter9.md:716` | `references/chapters/ch09-distributed-inference.md` |
| Сравнение схем развёртывания | `references/source-book/chapter9.md:734`, `references/source-book/chapter9.md:752`, `references/source-book/chapter9.md:793` | `references/chapters/ch09-distributed-inference.md` |
| Задача обучения, критический batch, состояние обучения | `references/source-book/chapter10.md:21`, `references/source-book/chapter10.md:59` | `references/chapters/ch10-training-systems.md` |
| Нижняя граница обучения по вычислениям и MFU | `references/source-book/chapter10.md:116` | `references/chapters/ch10-training-systems.md` |
| ZeRO/FSDP, повторное вычисление, выгрузка | `references/source-book/chapter10.md:151`, `references/source-book/chapter10.md:205`, `references/source-book/chapter10.md:237` | `references/chapters/ch10-training-systems.md` |
| Micro-batch и конвейер: fill–drain, 1F1B, пузыри | `references/source-book/chapter10.md:301`, `references/source-book/chapter10.md:322` | `references/chapters/ch10-training-systems.md` |
| Перекрытие коммуникаций; MoE и длинный контекст в обучении | `references/source-book/chapter10.md:408`, `references/source-book/chapter10.md:438` | `references/chapters/ch10-training-systems.md` |
| Checkpoint: состав, асинхронное сохранение, период | `references/source-book/chapter10.md:507`, `references/source-book/chapter10.md:527`, `references/source-book/chapter10.md:553` | `references/chapters/ch10-training-systems.md` |
| Ввод данных и отстающие узлы | `references/source-book/chapter10.md:491`, `references/source-book/chapter10.md:600` | `references/chapters/ch10-training-systems.md` |
| Системы RL: ресурсы этапов, синхронизация весов, асинхронность | `references/source-book/chapter10.md:629`, `references/source-book/chapter10.md:645`, `references/source-book/chapter10.md:685` | `references/chapters/ch10-training-systems.md` |
| Срок обучения и выбор оборудования | `references/source-book/chapter10.md:765`, `references/source-book/chapter10.md:798`, `references/source-book/chapter10.md:814` | `references/chapters/ch10-training-systems.md` |
| Потребность агентной задачи в CPU и времени среды | `references/source-book/chapter11.md:33`, `references/source-book/chapter11.md:62`, `references/source-book/chapter11.md:101` | `references/chapters/ch11-resource-scheduling.md` |
| Изоляция и жизненный цикл сред выполнения | `references/source-book/chapter11.md:121`, `references/source-book/chapter11.md:203`, `references/source-book/chapter11.md:235` | `references/chapters/ch11-resource-scheduling.md` |
| Общий пул: групповое выделение, приоритеты, RL | `references/source-book/chapter11.md:294`, `references/source-book/chapter11.md:322`, `references/source-book/chapter11.md:370` | `references/chapters/ch11-resource-scheduling.md` |
| Сервис модели: бюджет рассуждения, кэш префикса, собственный сервис против API и подписки | `references/source-book/chapter11.md:462`, `references/source-book/chapter11.md:488`, `references/source-book/chapter11.md:552` | `references/chapters/ch11-resource-scheduling.md` |
| Восстановление и полная стоимость успешной задачи | `references/source-book/chapter11.md:582`, `references/source-book/chapter11.md:625`, `references/source-book/chapter11.md:679` | `references/chapters/ch11-resource-scheduling.md` |
| Полное время взаимодействия: изображение, речь, Computer Use | `references/source-book/chapter12.md:13`, `references/source-book/chapter12.md:70`, `references/source-book/chapter12.md:114` | `references/chapters/ch12-edge-cloud.md` |
| Конечное устройство и локальное развёртывание | `references/source-book/chapter12.md:140`, `references/source-book/chapter12.md:156` | `references/chapters/ch12-edge-cloud.md` |
| Распределение работы между устройствами и миграция сеанса | `references/source-book/chapter12.md:216`, `references/source-book/chapter12.md:259`, `references/source-book/chapter12.md:291` | `references/chapters/ch12-edge-cloud.md` |
| Глобальная сеть: соединение, окно, потери пакетов | `references/source-book/chapter12.md:319`, `references/source-book/chapter12.md:329`, `references/source-book/chapter12.md:359` | `references/chapters/ch12-edge-cloud.md` |
| Беспроводная связь, сроки, два пути | `references/source-book/chapter12.md:462`, `references/source-book/chapter12.md:486`, `references/source-book/chapter12.md:500` | `references/chapters/ch12-edge-cloud.md` |
| Выбор между устройством, периферией и облаком | `references/source-book/chapter12.md:538`, `references/source-book/chapter12.md:556`, `references/source-book/chapter12.md:630` | `references/chapters/ch12-edge-cloud.md` |
| Заблуждения и ловушки глав | `references/source-book/chapter1.md:438`, `references/source-book/chapter5.md:837`, `references/source-book/chapter10.md:854` | `references/fallacies.md` |
| Краткие справочники и справочные таблицы чисел | `references/source-book/chapter1.md:450`, `references/source-book/chapter2.md:935` | `references/numbers.md` |

Остальные разделы заблуждений: `references/source-book/chapter2.md:903`, `references/source-book/chapter3.md:701`, раздел 7.6.5 — `references/source-book/chapter7.md:1009`. Порядок анализа на одной странице — `references/cheatsheet.md`, термины — `references/glossary.md`.

## Ссылки сносок книги

Сноски и ссылки в тексте книги ведут на материалы репозитория оригинала относительными путями: `../calculations/`, `../experiments/`, `../references/`, `../research/`, а также `../case-studies/` и `../archive/`. В переводе и в этом скилле этих файлов нет: такой путь — не локальный файл и не якорь на книгу. Каталог `../references/` оригинала — это не `references/` скилла.

Главы оригинала лежат в подкаталоге `book` его репозитория, поэтому `../` означает корень репозитория оригинала. Путь разрешается приписыванием префикса:

```text
https://github.com/bojieli/ai-infra-book/tree/56ecb425b07ea6d16e891cba87bf7db416927d09/
```

Например, `../calculations/results/decode-budget-base.md` из сноски главы 1 — это `https://github.com/bojieli/ai-infra-book/tree/56ecb425b07ea6d16e891cba87bf7db416927d09/calculations/results/decode-budget-base.md` (для файла GitHub откроет ту же ревизию через `/blob/`). Берите именно этот коммит: на других коммитах результаты и эксперименты могут отличаться или отсутствовать.

| Путь в сноске | Что там |
|---|---|
| `../calculations/` | калькуляторы автора, конфиги (`calculations/configs/hardware.json` — источник `data/hardware.json`) и результаты расчётов |
| `../experiments/` | эксперименты по главам: инструкции запуска, входные условия, зафиксированные результаты |
| `../references/` | научные статьи и техническая документация, на которые ссылается книга |
| `../research/` | исследовательские отчёты и проверки автора |
| `../case-studies/` | разборы отдельных случаев, например прерываний запусков RL |
| `../archive/` | архив планов глав |

Ссылки вида `chNN/figure-*.svg` — рисунки; в перевод и в скилл они не включены, подпись рисунка остаётся в тексте.

Для точного воспроизведения сценария книги нужен checkout оригинала на пине и команда автора из `calculations/`: калькулятор скилла `scripts/calc.py` переносит общие формулы, а не все сценарные команды автора.

## Drift gate

Книга и снимок железа зафиксированы на дату книги: текст — сентябрь 2026, `data/hardware.json` — снимок оригинала на коммите `56ecb425` от 2026-09-26 (151 запись, у каждого числа `source_id` и `locator`). Не используйте их как подтверждение текущего состояния рынка. Перед утверждением о любом из пунктов ниже:

| Что | Что проверить | Где |
|---|---|---|
| Цены: час ускорителя, облачные тарифы, $ за миллион токенов API, подписки | текущий прайс на дату ответа; в калькулятор цены передаются только аргументом (`--price-per-hour`, `--input-price`, `--output-price`, `--cache-read-price`, `--cache-write-price`) | прайс провайдера |
| Спецификации ускорителя: ёмкость, пропускная способность памяти, пик нужной точности, межсоединение | текущая спецификация производителя; значения снимка переопределяются `--memory`, `--bandwidth`, `--peak-tflops` | документация производителя; `python3 scripts/calc.py device --device <id>` печатает дату снимка и все пики по точностям, источник каждого числа — `source_id` и `locator` в `data/hardware.json` |
| Версии и конфигурации моделей | актуальный `config.json` нужной модели из официального репозитория; конфиги `scripts/tests/fixtures/configs/` — только эталоны тестов на дату книги | карточка и `config.json` модели |
| Версии движков, библиотек и драйверов (vLLM, NCCL и другие) | числа измерений книги верны для её версий и железа; перенос на другие версии — гипотеза до собственного измерения | заметки о выпуске, свой замер |
| Сеть и облако: RTT, полоса, квоты и лимиты API | значения своей среды и своих квот | измерение, документация сервиса |

Порядок:

1. Найти текущий первичный источник: спецификацию производителя, прайс, карточку и `config.json` модели.
2. Указать дату или версию источника.
3. Отделить факт от проектного вывода; число книги называть числом книги с её датой.
4. Если проверить нельзя — назвать значение неизвестным и предложить измерение.
