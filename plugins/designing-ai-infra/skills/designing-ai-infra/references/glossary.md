# Глоссарий

Термины в том смысле, в каком их используют книга и конспекты. Для каждого — форма в переводе, короткое определение и где читать: конспект главы и раздел книги. Идентификаторы и аббревиатуры оставлены в исходной форме, как в переводе.

## Единицы и счёт

**B, bit** — байт и бит: 1 byte = 8 bit. Скорость канала в bit/s делится на 8, чтобы получить B/s: 400 Gbit/s = 50 GB/s. `references/chapters/ch01-ai-infra-basics.md`, `references/source-book/chapter1.md:197`.

**GB и GiB** — десятичная и двоичная единицы ёмкости: KB, MB, GB, TB = 10³, 10⁶, 10⁹, 10¹² байт; KiB, MiB, GiB = 2¹⁰, 2²⁰, 2³⁰ байт. 141,11 GB = 131,42 GiB. Номинальная ёмкость ускорителя указана в GB производителя; вычитать веса из ёмкости можно только после приведения к одной единице. `calc.py units`. `references/source-book/chapter1.md:109`, `references/source-book/chapter1.md:197`.

**FLOP, FLOPs, FLOP/s** — одна операция с плавающей точкой; общее число операций; операций в секунду. GFLOPs и TFLOPs — миллиард и триллион операций (объём работы), GFLOP/s и TFLOP/s — скорость. В матричном умножении умножение и сложение считаются как 2 FLOP. В тексте скилла объём работы пишется GFLOPs/TFLOPs, скорость — TFLOP/s; калькулятор печатает объём с единицей FLOP (`decode_step_flops`), скорость — FLOP/s (`peak_flops`). `references/source-book/chapter1.md:160`.

**Пик нужной точности** — пиковая матричная производительность задана для точности входа (BF16, FP8, FP4), точности накопления (FP32 или FP16) и вида вычислений (плотные или со структурной разреженностью). Пик FP8 нельзя подставлять в задачу BF16, разреженный — в плотную. В калькуляторе — `--precision`, `--accumulator`, `--sparsity`. `references/chapters/ch01-ai-infra-basics.md`, `references/source-book/chapter1.md:160`.

**BF16, FP16, FP8, FP4, int8** — форматы чисел: 2, 2, 1 и 0,5 байта на значение (int8 — 1 байт). BF16 имеет диапазон FP32 и короткую мантиссу, FP16 требует масштабирования функции потерь при обучении, FP8 — блочного масштабирования. Квантованные веса занимают больше, чем произведение числа параметров на байты формата: добавляются scale и части в высокой точности. `references/chapters/ch10-training-systems.md`, `references/source-book/chapter10.md:59`, `references/source-book/chapter4.md:337`, `references/source-book/chapter8.md:386`.

## Модель времени и метрики

**M, F, R; M_cap, Π, β** — сколько байт хранить, сколько FLOP выполнить и сколько байт прочитать-записать; ёмкость, вычислительная пропускная способность и пропускная способность интерфейса ускорителя. M и R считаются отдельно: данные сохраняют один раз, а читают многократно. `references/chapters/ch01-ai-infra-basics.md`, `references/source-book/chapter1.md:160`.

**Нижняя граница времени** — `T ≥ max(F/Π, R/β)` при пиковых Π и β и полном перекрытии вычислений и чтения; физический предел, быстрее которого программная организация работать не может. Калькулятор помечает такие значения «(нижняя граница)». `references/source-book/chapter1.md:271`.

**MFU, MBU** — model FLOPs utilization («коэффициент использования FLOPs модели») `F/(Π·T)` и memory bandwidth utilization («коэффициент использования пропускной способности памяти») `R/(β·T)` по измеренному T; оба не больше 1. MFU обучения берётся с суммарным пиком всех GPU в знаменателе. `references/chapters/ch01-ai-infra-basics.md`, `references/chapters/ch10-training-systems.md`, `references/source-book/chapter1.md:160`, `references/source-book/chapter10.md:116`.

**Арифметическая интенсивность** — FLOP на прочитанный байт, `I = F/R`. Если I меньше `Π/β`, дольше чтение; если больше — вычисления. `calc.py roofline` печатает `arithmetic_intensity`. `references/source-book/chapter1.md:271`.

**Roofline** — модель «линия крыши»: достижимая производительность `≤ min(P, R·I)` (обозначения главы 4: P — пик, R — пропускная способность памяти, I — FLOP на байт; в главе 1 те же величины — Π и β); наклонная часть — предел памяти, горизонтальная — предел матричного блока. `references/chapters/ch04-accelerators.md`, `references/source-book/chapter4.md:972`.

**Ridge point (точка пересечения)** — интенсивность `I* = P/R` (в обозначениях главы 1 — `Π/β`), где пересекаются две линии Roofline; выше неё узкое место — вычисления. Калькулятор печатает `ridge_point` в FLOP/B; в книге и конспектах это «точка пересечения». `references/source-book/chapter4.md:972`.

**B\* (точка перехода по batch)** — размер batch, при котором время вычислений догоняет время чтения весов: `B* = b_W·Π/(2β)`, для H100 и 1 байта на параметр ≈ 147,7. Не учитывает KV: с длинным контекстом чтение KV догоняет веса раньше (`b_KV`). Оба порога с KV — `calc.py batch-threshold`. `references/chapters/ch01-ai-infra-basics.md`, `references/chapters/ch08-inference-optimization.md`, `references/source-book/chapter1.md:271`, `references/source-book/chapter8.md:104`.

**TTFT** — время до первого выходного токена: `TTFT = t_1 − t_a` от поступления запроса; включает очередь и prefill. `references/chapters/ch03-workloads.md`, `references/chapters/ch08-inference-optimization.md`, `references/source-book/chapter3.md:29`, `references/source-book/chapter8.md:82`.

**TPOT** — время на выходной токен (time per output token): наблюдаемый клиентом средний интервал между выходами; в измерениях главы 1 — интервал от первого до последнего выхода, делённый на число интервалов, медиана. `references/source-book/chapter1.md:333`.

**ITL** — интервал между соседними выходными токенами `ITL_j = t_{j+1} − t_j`; полное время запроса `T_request = TTFT + Σ ITL_j`. Максимальный ITL — то, что ломают длинный prefill соседа и крупный бюджет токенов. `references/chapters/ch08-inference-optimization.md`, `references/source-book/chapter8.md:82`.

**p95** — 95-й процентиль методом ближайшего ранга: `⌈0,95·n⌉`-е значение по возрастанию. `references/chapters/ch03-workloads.md`, `references/source-book/chapter3.md:29`.

**SLO** — целевые пределы сервиса для TTFT, ITL и полного времени. `references/chapters/ch08-inference-optimization.md`, `references/source-book/chapter8.md:82`.

**Эффективная пропускная способность (goodput)** — число правильных результатов, завершённых в срок, в единицу времени; не то же, что пропускная способность завершения. `references/chapters/ch08-inference-optimization.md`, `references/source-book/chapter8.md:616`.

**Площадь состояния** — ёмкость, умноженная на время её занятости, `A_M = ∫ M(t) dt`: 1 GiB на 10 s — 10 GiB·s. Ожидание инструмента тоже занимает ёмкость. `references/chapters/ch03-workloads.md`, `references/source-book/chapter3.md:29`.

**Закон Амдала** — ускорение доли f времени в s раз даёт `1/((1 − f) + f/s)`, не больше `1/(1 − f)`. `references/chapters/ch01-ai-infra-basics.md`, `references/source-book/chapter1.md:109`, `references/source-book/chapter5.md:934`.

**Закон Литтла** — среднее число в системе равно интенсивности поступления, умноженной на время пребывания: `L = λ·W`. Применяется к запросам, транзакциям памяти в полёте и слотам сетевых запросов. `calc.py queueing`. `references/chapters/ch08-inference-optimization.md`, `references/source-book/chapter8.md:598`, `references/source-book/chapter4.md:443`.

## Модель и её состояние

**Prefill** — этап, на котором модель обрабатывает весь известный вход и выдаёт первый токен; крупные матрицы, веса переиспользуются многими строками, обычно упор в вычисления. `references/chapters/ch02-model-architecture.md`, `references/source-book/chapter2.md:220`, `references/source-book/chapter3.md:13`.

**Decode** — пошаговая генерация по одному токену на вызов; каждый шаг читает веса и KV всего контекста, обычно упор в пропускную способность памяти. `references/source-book/chapter2.md:220`, `references/source-book/chapter3.md:13`.

**KV-кэш (KV)** — сохранённые K и V прежних токенов, заменяющие их повторное вычисление. Для слоёв с полным KV на токен `c_KV = 2·L·n_KV·d_h·b`: у Qwen3-8B в BF16 — 144 KiB на токен. Растёт с каждым токеном и не делится между запросами, кроме общего префикса. `calc.py model` → `kv_bytes_per_token`. `references/chapters/ch02-model-architecture.md`, `references/source-book/chapter2.md:220`, `references/source-book/chapter2.md:454`.

**MHA, GQA, MQA** — внимание со своим KV у каждой головы запроса; с общим KV у группы голов; с одним KV на все головы. Меньше групп KV — меньше ёмкость и чтение KV, но FLOP внимания по-прежнему суммируются по головам запросов. `references/chapters/ch02-model-architecture.md`, `references/source-book/chapter2.md:312`.

**MLA** — хранение контекста как низкоразмерной скрытой переменной; компактный кэш преобразует запрос на каждом шаге, развёрнутый читает K и V каждой головы. Объём зависит от пути выполнения, а не только от конфигурации. `references/chapters/ch02-model-architecture.md`, `references/source-book/chapter2.md:334`.

**Окно, линейное и гибридное внимание** — локальное окно хранит KV последних токенов; линейное внимание держит состояние фиксированного размера; гибридное сочетает фиксированную часть с растущей на токен. Незнакомую архитектуру калькулятор не считает как плотную: печатает «не вычисляется» и перечисляет поля. `references/chapters/ch02-model-architecture.md`, `references/source-book/chapter2.md:360`, `references/source-book/chapter2.md:396`, `references/source-book/chapter2.md:418`.

**MoE (смесь экспертов, Mixture of Experts)** — маршрутизатор выбирает для токена k_top из E экспертов: хранятся все эксперты, считаются выбранные, а batch читает объединение затронутых экспертов. «Активные параметры» — оценка для одного токена. `references/chapters/ch02-model-architecture.md`, `references/source-book/chapter2.md:532`, `references/source-book/chapter2.md:542`.

**Общий эксперт** — эксперт, который обрабатывает все строки, в дополнение к маршрутизируемым. `references/source-book/chapter2.md:573`.

**MTP** — предсказание нескольких токенов вспомогательным модулем; кандидаты проверяет целевая модель, как при спекулятивном декодировании. `references/chapters/ch02-model-architecture.md`, `references/source-book/chapter2.md:621`.

**CED** — причинный кодировщик-декодировщик DeepSeek V4.1 Flash: большинство входных токенов не проходит основную часть декодировщика, а каждый генерируемый токен проходит все слои. Экономию prefill нельзя переносить на decode. `references/chapters/ch02-model-architecture.md`, `references/source-book/chapter2.md:454`.

**Engram** — модуль V4.1 Flash, который по n-грамме получает дополнительное представление из таблицы; параметры таблицы считаются отдельно от основной части. `references/source-book/chapter2.md:3`, `references/source-book/chapter6.md:1021`.

**6ND** — оценка FLOPs обучения: 6 × параметры × токены. Поэлементный расчёт учитывает внимание и отличается на длинных последовательностях. `calc.py training` печатает обе оценки. `references/chapters/ch03-workloads.md`, `references/source-book/chapter3.md:442`.

## Сервинг на одном экземпляре

**Экземпляр инференса** — набор из одного или нескольких ускорителей, который независимо обрабатывает запросы. `references/chapters/ch01-ai-infra-basics.md`, `references/source-book/chapter1.md:57`.

**Batch** — токены, обрабатываемые одним выполнением на ускорителе; запросы batch делят одно чтение весов, но каждый читает свой KV. `references/chapters/ch08-inference-optimization.md`, `references/source-book/chapter8.md:104`.

**Непрерывная пакетная обработка (continuous batching)** — перераспределение активных запросов на границах итераций с приёмом новых сразу после освобождения позиций; фиксированная пакетная обработка принимает и завершает запросы группой. `references/chapters/ch08-inference-optimization.md`, `references/source-book/chapter8.md:165`.

**Блочный prefill (chunked prefill)** — разбиение длинного prefill на блоки, чтобы он не останавливал decode текущих запросов; крупный блок — меньше лишних чтений весов, мелкий — меньше максимальный ITL. `references/source-book/chapter8.md:191`, `references/source-book/chapter9.md:154`.

**Бюджет токенов** — предел новых токенов за раунд планировщика; выбирается от допустимого ITL. `references/source-book/chapter8.md:218`.

**Страничное распределение KV (paged KV, PagedAttention)** — логические блоки позиций, физические блоки памяти и таблица блоков на запрос; потери — только в последнем блоке. Управляет размещением состояния, а не вычислением внимания. `references/chapters/ch08-inference-optimization.md`, `references/source-book/chapter8.md:232`.

**Копирование при записи (copy-on-write)** — общий блок KV копируется только тогда, когда ветвь начинает писать в него своё. Освободить блок можно, когда на него нет ссылок и уже отправленные операции ускорителя к нему не обращаются. `references/source-book/chapter8.md:268`.

**Повторное использование префикса (кэш префикса)** — KV общего префикса переживает запрос; при полном совпадении префикса, модели, позиционного кодирования и адаптера prefill продолжается с конца префикса. `references/chapters/ch08-inference-optimization.md`, `references/chapters/ch11-resource-scheduling.md`, `references/source-book/chapter8.md:310`, `references/source-book/chapter11.md:488`.

**Вытеснение** — освобождение KV запроса при нехватке памяти ценой повторного вычисления позже. `references/source-book/chapter8.md:268`, `references/source-book/chapter8.md:354`.

**Сжатие и выгрузка** — квантование весов и KV уменьшает ёмкость и чтение, но добавляет время преобразования; выгрузка держит веса или KV в более медленной памяти и переносит по требованию. `references/chapters/ch08-inference-optimization.md`, `references/source-book/chapter8.md:398`, `references/source-book/chapter8.md:432`.

**Спекулятивное декодирование** — черновик из нескольких токенов для одного запроса проверяется целевой моделью за одно вычисление; при правильном принятии распределение выхода не меняется. Выигрыш есть, пока время раунда меньше `E[N] × T_plain`, где `E[N]` — ожидаемое число токенов за раунд. `calc.py speculative`. `references/chapters/ch08-inference-optimization.md`, `references/source-book/chapter8.md:504`, `references/source-book/chapter8.md:536`.

**LoRA** — поправка весов как произведение двух малых матриц над общими базовыми весами; каждый адаптер занимает свою ёмкость. `references/source-book/chapter8.md:104`.

## Параллелизм и распределённый инференс

**DP, TP, SP, CP, PP, EP** — параллелизм по данным (копии модели, разные образцы), тензорный (части матриц слоя; частичные суммы складываются при разбиении измерения редукции — уточнение русского издания, `references/source-book/chapter6.md:116`), по последовательности (позиции в поточечных операторах внутри группы TP), контекста (позиции одной последовательности во внимании), конвейерный (группы слоёв) и экспертный (части набора экспертов). Разбиваемые измерения: B, H, S, S, L, E. `references/chapters/ch06-supernodes.md`, `references/source-book/chapter6.md:80`, `references/source-book/chapter6.md:134`.

**Предел KV-голов при TP** — если карт TP больше, чем KV-голов, KV-кэш дублируется. `references/source-book/chapter6.md:148`. Правило для GQA/MHA; компактный кэш MLA хранит одну латентную переменную на токен и слой, общую для всех голов (`references/source-book/chapter9.md:100`), поэтому при TP не делится вовсе (вывод из 2.3.2 и 6.2.2, числового примера TP для MLA в книге нет), а `num_key_value_heads` в config MLA — не число разделимых голов.

**Ring Attention** — способ планирования передачи K и V по кольцу; параллелизм контекста — способ разделения работы, это не синонимы. `references/source-book/chapter6.md:272`.

**Dispatch и combine** — два обмена All-to-All при EP: отправка входа токена к картам выбранных экспертов и возврат результатов со взвешенным сложением. `references/source-book/chapter6.md:329`.

**Micro-batch** — часть batch, которую конвейер обрабатывает как независимую единицу; утилизация конвейера `b/(q + b − 1)`. `references/source-book/chapter6.md:300`, `references/source-book/chapter10.md:307`.

**PD-разделение (разделение PD, Prefill–Decode)** — prefill и decode выполняются на разных ресурсах со своими очередями и batch; KV передаётся от P к D один раз. В однородном кластере, по автору, GPU-секунды на запрос не меняются и пропускная способность не растёт; выгода — изоляция очередей и стабильный ITL. Русское издание ограничивает вывод условиями примера: при другом batch или параллелизме этапов GPU-секунды оцениваются заново, goodput при SLO — отдельно от throughput. `references/chapters/ch09-distributed-inference.md`, `references/source-book/chapter9.md:3`, `references/source-book/chapter9.md:90`, `references/source-book/chapter9.md:154`.

**AF-разделение (разделение AF)** — внимание и сеть прямого распространения каждого слоя, включая экспертов, выполняются на разных ресурсах; активации передаются туда и обратно на каждом слое и шаге. `references/chapters/ch09-distributed-inference.md`, `references/source-book/chapter9.md:3`, `references/source-book/chapter9.md:304`.

**Реплика и общий KV** — полная копия модели и её состояния, принимающая запросы независимо; общий KV позволяет другим экземплярам использовать оставленное состояние. `references/source-book/chapter9.md:63`.

**Маршрутизация с учётом кэша** — выбор экземпляра с учётом того, где уже лежит KV префикса и какая там очередь: `T_first = max(Q, R) + C` (очередь, получение префикса, вычисление); экземпляр с кэшем, но длинной очередью, может проиграть. `references/chapters/ch09-distributed-inference.md`, `references/source-book/chapter9.md:611`.

## Коммуникации, суперузлы и сеть

**AllReduce, ReduceScatter, AllGather, All-to-All** — полный результат у всех; свой сегмент суммы; весь тензор из сегментов; каждая карта получает то, что ей прислали. AllReduce = ReduceScatter + AllGather. `references/chapters/ch06-supernodes.md`, `references/source-book/chapter6.md:471`.

**α и кольцевой AllReduce** — α — время запуска одного раунда; кольцо: `T_ring = 2(n−1)α + 2(n−1)M/(nB)`. Для малых сообщений доминирует α и полоса почти не помогает; для крупных — наоборот. `calc.py ring`; кольцо против дерева и точка равенства — `calc.py allreduce`. `references/source-book/chapter6.md:490`.

**Суперузел** — группа ускорителей, тесно взаимодействующих через высокоскоростной интерконнект; может занимать несколько серверов или вычислительных лотков. `references/chapters/ch06-supernodes.md`, `references/source-book/chapter1.md:57`, `references/source-book/chapter6.md:13`.

**Scale-up и scale-out** — интерконнект внутри суперузла (например, NVLink) для совместных вычислений с малыми издержками; сетевые карты и коммутируемая сеть между суперузлами. `references/source-book/chapter1.md:57`.

**Unified Bus (UB)** — межсоединение Huawei, дающее устройствам прямой доступ к памяти других устройств без промежуточных уровней передачи сообщений. `references/source-book/chapter1.md:409`, `references/source-book/chapter6.md:752`. В главе 4 то же сокращение UB означает другое — единый буфер (Unified Buffer) блока Vector в Ascend DaVinci (`references/source-book/chapter4.md:83`).

**Пул памяти** — заимствование памяти других устройств суперузла при нехватке локальной ёмкости. `references/source-book/chapter6.md:839`.

**Rail (рельс), многорельсовая топология** — путь между сетевыми картами с одинаковыми номерами на разных серверах через один коммутатор; i-я NIC каждого сервера подключена к i-му leaf-коммутатору. Обмен пар с одинаковым `i mod 8` остаётся в одной rail. `references/chapters/ch07-datacenter-network.md`, `references/source-book/chapter7.md:17`, `references/source-book/chapter7.md:299`.

**Сеть Clos, переподписка, бисекционная пропускная способность** — многоуровневая сеть из leaf- и spine-коммутаторов; коэффициент `d/u` — отношение пропускной способности нисходящих и восходящих подключений leaf-коммутатора; совокупная пропускная способность соединений между двумя половинами кластера с одинаковым числом конечных узлов. `references/source-book/chapter7.md:17`, `references/source-book/chapter7.md:41`.

**Односторонние Read/Write (RDMA) и Load/Store** — передача в заранее разрешённую удалённую память без участия программы получателя; двусторонние сообщения требуют, чтобы получатель заранее выставил буфер. Передача — это две операции: положить данные и сообщить о готовности. Load/Store — инструкции процессора, зависимости отслеживает аппаратура; асинхронные Read/Write — запрос с адресом и длиной в очереди, завершение приходит событием. По названию интерфейса нельзя судить о когерентности кэшей. `references/chapters/ch07-datacenter-network.md`, `references/source-book/chapter7.md:319`, `references/source-book/chapter7.md:367`.

**Incast** — несколько отправителей одновременно передают в один выходной канал; поступление `(N − 1)B` переполняет буфер за микросекунды. `references/chapters/ch07-datacenter-network.md`, `references/source-book/chapter7.md:700`.

**PFC, ECN/DCQCN** — управление потоком на уровне канала (останавливает соседа за один переход) и сквозное управление перегрузкой (снижает скорость источника). PFC при циклической зависимости даёт взаимную блокировку. `references/source-book/chapter7.md:700`, `references/source-book/chapter7.md:812`.

**ECMP** — многопутевая маршрутизация с равной стоимостью: все пакеты одного потока идут одним маршрутом, разные потоки распределяются по восходящим каналам хешем заголовка; два потока могут попасть на один канал. `references/source-book/chapter7.md:758`.

## Ускоритель и операторы

**HBM** — видеопамять ускорителя с высокой пропускной способностью; её β задаёт нижнюю границу чтения весов и KV. `references/chapters/ch04-accelerators.md`, `references/source-book/chapter4.md:379`.

**Kernel, launch, stream, event** — программа, выполняемая ускорителем; её асинхронная отправка хостом; очередь заданий; отметка завершения. Время отправки и время готовности результата различаются. `references/chapters/ch05-operators-runtime.md`, `references/source-book/chapter5.md:27`, `references/source-book/chapter5.md:71`.

**SM** — вычислительный блок GPU NVIDIA, на котором выполняются блоки kernel. `references/source-book/chapter4.md:71`, `references/source-book/chapter5.md:114`.

**Тайл и слияние операторов** — разбиение матрицы на блоки, помещающиеся во внутрикристальную память; объединение соседних операторов, чтобы промежуточный тензор не уходил во внешнюю память. `references/chapters/ch05-operators-runtime.md`, `references/source-book/chapter5.md:184`, `references/source-book/chapter5.md:345`.

**FlashAttention** — поблочное вычисление внимания с онлайн-Softmax: полные матрицы оценок S и P не хранятся, вместо них держатся максимум, сумма экспонент и взвешенная сумма V. `references/chapters/ch05-operators-runtime.md`, `references/source-book/chapter5.md:403`.

**Двойная буферизация** — загрузка следующего блока, пока обрабатывается текущий. `references/source-book/chapter4.md:523`, `references/source-book/chapter5.md:371`.

**CUDA Graph** — заранее захваченная последовательность kernel, отправляемая одной операцией; экономит время хоста, но требует фиксированных форм и копирования входа. `references/source-book/chapter5.md:820`.

**Persistent kernel** — kernel, который постоянно работает на ускорителе, берёт задачи из очереди и проверяет готовность по флагам: следующий оператор стартует по готовому блоку, а не по завершению всего kernel. `references/source-book/chapter5.md:904`.

## Обучение

**Состояние обучения** — веса, градиенты, основные веса FP32 и два момента Adam: 16 байт на параметр при градиентах BF16 (18 при градиентах FP32). В MoE — по всем экспертам. `references/chapters/ch10-training-systems.md`, `references/source-book/chapter10.md:59`.

**ZeRO, FSDP** — шардирование состояния между GPU группы DP: этап 1 — основные веса и моменты, этап 2 — ещё градиенты, этап 3 — ещё веса; FSDP организует выполнение по тому же принципу. Постоянное состояние и пик памяти уменьшаются в разной степени. `calc.py training --dp`; по компонентам — `calc.py training-state --stage`. `references/source-book/chapter10.md:151`.

**Повторное вычисление и выгрузка** — пересчёт активаций в обратном проходе вместо хранения; перенос состояния между GPU и CPU. `references/source-book/chapter10.md:211`, `references/source-book/chapter10.md:243`.

**Fill–drain и 1F1B** — расписания конвейера: сначала все прямые проходы, потом все обратные; после прогрева — чередование одного прямого и одного обратного, активации ранних micro-batch освобождаются раньше. Пузырь у них одинаковый, различается резидентность активаций. `references/chapters/ch10-training-systems.md`, `references/source-book/chapter10.md:328`.

**Пузырь конвейера** — простой этапов при заполнении и опустошении конвейера; доля и время — `calc.py pipeline`. `references/source-book/chapter6.md:300`, `references/source-book/chapter10.md:328`.

**Checkpoint** — сохранённое состояние, достаточное для продолжения обучения: веса, оба момента, номер шага, состояние learning rate, случайное состояние, состав следующего batch и ещё не упакованные токены. Асинхронный снимок пригоден для восстановления только после записи и фиксации. `references/chapters/ch10-training-systems.md`, `references/source-book/chapter10.md:513`, `references/source-book/chapter10.md:533`.

**Период checkpoint** — интервал τ, минимизирующий долю потерь `c/τ + λτ/2 + λr`; оптимум первого порядка `τ* = √(2c/λ)`. `calc.py checkpoint`. `references/source-book/chapter10.md:559`.

**Отстающий узел** — медленный участник синхронного шага; шаг определяется максимумом, а не средним. `references/source-book/chapter10.md:606`.

**Критический batch (критический размер батча)** — размер batch, после которого добавление примеров лишь незначительно сокращает число шагов до того же значения функции потерь, а вычисления на шаг растут пропорционально; оценивается через масштаб шума градиента. Поэтому выше него добавленные карты при пропорциональном росте batch почти не сокращают срок. `references/chapters/ch10-training-systems.md`, `references/source-book/chapter10.md:21`, `references/source-book/chapter10.md:860`.

**Rollout** — генерация траекторий в RL, за которой идут проверка средой, вознаграждение и обновление стратегии; единица работы — пригодный для обучения образец. `references/chapters/ch03-workloads.md`, `references/chapters/ch10-training-systems.md`, `references/source-book/chapter3.md:476`, `references/source-book/chapter10.md:635`.

## Среды, стоимость, устройство и облако

**microVM** — лёгкая виртуальная машина как граница изоляции среды агента, между контейнером и полной VM. `references/chapters/ch11-resource-scheduling.md`, `references/source-book/chapter11.md:121`.

**Групповое выделение** — многокарточное задание запрашивает тип ускорителя, видеопамять, сопутствующие CPU и память узла, размещение под схему обмена и одновременный запуск всех процессов; отсюда фрагментация пула. `references/chapters/ch11-resource-scheduling.md`, `references/source-book/chapter11.md:294`.

**Стоимость успешной задачи** — затраты всех попыток (генерация, проверка, окружение, неудачи), делённые на число успешных задач. `calc.py cost` → `cost_per_accepted_task`. `references/chapters/ch11-resource-scheduling.md`, `references/source-book/chapter3.md:173`, `references/source-book/chapter11.md:625`.

**Полное время взаимодействия** — при последовательных этапах `T_serial = S_u/B_u + R + T_c + S_d/B_d`: загрузка, распространение туда и обратно, работа модели, выгрузка; время модели — лишь одно слагаемое. `calc.py edge` → `serial_seconds`. `references/chapters/ch12-edge-cloud.md`, `references/source-book/chapter12.md:11`.

**RTT** — время передачи туда и обратно; для небольших последовательных обменов доминирует над объёмом, делённым на пропускную способность. `references/source-book/chapter12.md:319`, `references/source-book/chapter12.md:329`.
