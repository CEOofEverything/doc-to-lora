# Eval‑pipeline fixes for the random‑repr branch (Qwen3‑4B + diverse_sft)

Контекст: запуск
```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled \
  uv run run_eval.py \
  --checkpoint_path train_outputs/runs/May08_08-51-19_parchiev-synthetic-data-0_c548e024/checkpoint-3000/pytorch_model.bin \
  --datasets diverse_sft --split validation \
  --max_ctx_chunk_len 6144 --eval_batch_size_gen 1 --max_val_samples_per_ds 100
```
Чекпоинт обучен с `D2L_USE_RANDOM_REPR=1` (`cli_args.yaml`: `ctx_encoder_type=early_exit`, `per_rank_gen=False`). На eval вылезла серия багов: random‑repr ветка везде ждёт packed‑геометрию из тренинга, а HF `generate()` подаёт обычные padded батчи без `position_ids`. Дополнительно — пайплайн eval'а для **base‑model** прогонял другую геометрию, чем D2L, и метрики были неэквивалентны.

В этом документе — **все правки, сделанные за сессию**, в порядке появления.

---

## Обязательное условие запуска

`D2L_USE_RANDOM_REPR` читается **во время import‑а** (`hypernet.py:68`):
```python
USE_RANDOM_REPR = os.environ.get("D2L_USE_RANDOM_REPR", "0").lower() in ("1", "true", "yes")
```
Флаг — компиляционный switch для архитектуры (`HyperLoRA.__init__` создаёт другой набор параметров, голова имеет другой out‑shape, `generate_weights` возвращает другой тип, на `down_proj` вешается другой forward). Чекпоинт random‑ветки в обычную сборку **не загрузится корректно**.

→ Запускать eval random‑чекпоинта надо так:
```bash
export D2L_USE_RANDOM_REPR=1
```
**до** любого `uv run run_eval.py`.

---

## Fix 1 — `ctx_position_ids is None` в `hypernet.generate_weights`

### Симптом
```
File "src/ctx_to_lora/modeling/hypernet.py", line 734, in generate_weights
    position_ids = ctx_position_ids.squeeze(0)
AttributeError: 'NoneType' object has no attribute 'squeeze'
```

### Корень
`hypernet.py:732-758` (ветка `if USE_RANDOM_REPR:`) безусловно делает
`ctx_position_ids.squeeze(0)`, чтобы из «нулей в `position_ids`» восстановить
границы чанков. Это работает только в packed‑геометрии (трейн).

На eval `generation_collator` (`data/collator.py:106-148`) кладёт в батч только
`ctx_ids`, `ctx_attn_mask`, `n_ctx_chunks` — **без** `ctx_position_ids`. Поле
существует только в packing‑коллаторе тренинга (`collator.py:50-55`).

### Правка — `src/ctx_to_lora/modeling/hypernet.py`, ~lines 730‑760

Разнесли random‑repr ветку на packed / unpacked:

```python
if USE_RANDOM_REPR:
    n_ctx = len(n_ctx_chunks)
    if ctx_position_ids is None:
        # eval/generation: ctx_ids = [n_chunks, padded_len], padded по ctx_attn_mask.
        # repr_seeds = sum валидных token id по чанкам, сгруппированных в parent‑ctx.
        chunk_seeds = (ctx_ids * ctx_attn_mask).sum(dim=1)            # [n_chunks]
        chunk_to_ctx = torch.repeat_interleave(
            torch.arange(n_ctx, device=ctx_ids.device), n_ctx_chunks,
        )
        self.repr_seeds = torch.zeros(n_ctx, dtype=ctx_ids.dtype, device=ctx_ids.device)
        self.repr_seeds.scatter_add_(0, chunk_to_ctx, chunk_seeds)
    else:
        # packed (train): прежняя логика
        position_ids = ctx_position_ids.squeeze(0)
        ctx_lens = position_ids[torch.where(position_ids == 0)[0][1:] - 1]
        ctx_lens = torch.cat((ctx_lens,
                              torch.tensor([position_ids[-1]], device=ctx_lens.device)))
        ctx_lens += 1
        tot_len = ctx_lens.sum().item()
        tot_chunks = n_ctx_chunks.sum().item()
        index = torch.repeat_interleave(
            torch.arange(n_ctx, device=ctx_ids.device),
            n_ctx_chunks, dim=0, output_size=tot_chunks,
        )
        index = torch.repeat_interleave(index, ctx_lens, dim=0, output_size=tot_len)
        self.repr_seeds = torch.zeros(n_ctx, dtype=ctx_ids.dtype, device=ctx_ids.device)
        self.repr_seeds.scatter_add_(0, index, ctx_ids.squeeze(0))
    self.generator = torch.Generator(device=ctx_ids.device)
```

Семантика идентична: `repr_seed[i] = sum(token_id для всех валидных позиций всех
чанков ctx i)`. Паддинг отбрасывается через `ctx_attn_mask`, поэтому одинаковый
документ в трейне и в eval даёт **один и тот же seed**, и random Perceiver
генерирует ту же матрицу.

---

## Fix 2 — `apply_random_repr` падает на None и не учитывает kv‑cache

### Симптом
```
File "src/ctx_to_lora/modeling/lora_layer.py", line 327, in apply_random_repr
    position_ids = position_ids.squeeze(0)
AttributeError: 'NoneType' object has no attribute 'squeeze'
```

### Корень
`apply_random_repr` (`lora_layer.py:313`) тоже ждёт packed `position_ids` и зашивает
`seq_lens`/`tot_len` через `partial(module.forward, ...)`. На eval:
- `position_ids` приходит `None` (HF не пробрасывает его в kwargs).
- HF.generate использует kv‑cache: первый шаг — полный prompt, последующие —
  один новый токен. То есть `x.shape[1]` меняется → `seq_lens` нельзя баковать
  один раз, надо считать на каждом forward.

### Правка — `src/ctx_to_lora/modeling/lora_layer.py:313`

`apply_random_repr` теперь имеет две ветки:

| режим | условие | как навешивает forward |
|---|---|---|
| **packed** (train) | `position_ids is not None` | как раньше: считает `seq_lens`/`tot_len` из нулей в `position_ids`, зашивает в `partial`. |
| **unpacked** (eval/generate) | `position_ids is None` | НЕ зашивает `seq_lens`/`tot_len`. Оборачивает `module.forward` замыканием `make_wrapper`, которое **на каждом вызове** берёт `x.size(1)` и строит `seq_lens=[seq_len]*tot_q`, `tot_len=seq_len*tot_q`. |

Допущения unpacked‑ветки:
- `bs == 1` (выполняется при `--eval_batch_size_gen 1`),
- одна «query» на контекст (`tot_q == n_ctx`, выставляется в `hypernet.py:1111-1120`).

Если поднимать `eval_batch_size_gen > 1` — потребуется ещё одна ветка, потому что
`random_repr_forward` принимает `[1, tot_len, d_in]`, а не `[bs, seq_len, d_in]`.

---

## Fix 3 — dtype mismatch BFloat16 × Float32 в `random_repr_forward`

### Симптом
```
File "src/ctx_to_lora/modeling/lora_layer.py", line 290, in random_repr_forward
    delta_x_masked = einsum(...)
RuntimeError: expected scalar type BFloat16 but found Float
```

### Корень
- `coeffs` (выход hypernet) — **bf16**, потому что `HyperLoRA.forward` обёрнут в
  `torch.autocast(..., dtype=torch.bfloat16)` (`hypernet.py:467`).
- `repr_A`, `repr_B` создаются с `dtype=coeffs.dtype` → bf16.
- `x` (активации Qwen3 на eval) — **fp32**, потому что `run_eval` ставит
  `eval_trainer_args["bf16"] = False` (`eval_utils.py:939`).

В `lora_forward_packed` (`lora_layer.py:55, 72`) этот случай уже обрабатывается
(`x = x.to(A.dtype)`); в `random_repr_forward` касинга не было.

### Правка — `src/ctx_to_lora/modeling/lora_layer.py`, в `random_repr_forward`

В начале функции и в return:
```python
compute_dtype = coeffs.dtype
x = x.to(compute_dtype)
...
delta_x = torch.zeros((1, tot_len, self.out_features),
                      device=x.device, dtype=compute_dtype)
...
return (base_out + delta_x.to(base_out.dtype)).to(base_out.dtype)
```

Симметрично с обычной (не‑random) веткой; на трейне эффекта нет (там везде
автокастится в bf16), на eval — устраняет mismatch.

---

## Fix 4 — `TypeError: multiple values for keyword 'seq_lens'`

### Симптом
```
File "src/ctx_to_lora/modeling/lora_layer.py", line 391, in fn
    return bound_fn(...)
TypeError: functools.partial(<random_repr_forward>, self=lora.Linear(...),
        lora_dropout_p=0.0, scaling=45.25, n_queries=tensor([1]), tot_q=1,
        coeffs=..., n_ctx_chunks=..., repr_seeds=..., generator=...)
        got multiple values for keyword argument 'seq_lens'
```

### Корень
`apply_random_repr` вызывается **на каждый batch** eval‑predict’а из
`hypernet.py:1122-1132`. После первого batch `module.forward` уже обёрнут моим
`make_wrapper`. Второй вызов оборачивает обёрнутую обёртку: внешний `fn`
прокидывает `seq_lens` через kwarg → внутренний `fn` тоже пытается → конфликт.

В `hypernet.py:1019-1028` есть `model.reset()`, но между батчами он не зовётся.
В `_init_model` через `patch_lora_forward()` сохраняется `module.forward_orig` —
но это **сырой PEFT forward**, а не patched‑partial с `self`, `lora_dropout_p`,
`scaling`. Откатиться к нему нельзя: тогда наш `partial(peft_forward,
n_queries=..., coeffs=..., …)` уйдёт в чистый PEFT, который пробросит наши
LoRA‑kwargs в `nn.Linear`.

Промежуточная (моя) первая попытка с откатом к `forward_orig` именно поэтому
давала следующий креш в `peft/tuners/lora/layer.py:706` (`self.base_layer(x, *args,
**kwargs)` с неизвестными kwargs).

### Правка — два файла

#### `src/ctx_to_lora/modeling/hypernet.py:603-617` (в `patch_lora_forward`)

Снапшотим базовый patched‑partial в `module.forward_lora_base`:
```python
if getattr(module, "patched_forward", False):
    continue
module.forward_orig = module.forward                   # raw PEFT (для model.reset())
module.patched_forward = True
module.forward = partial(
    forward_fn, self=module,
    lora_dropout_p=self.peft_config.lora_dropout,
    scaling=self.peft_config.lora_alpha,
)
module.forward_lora_base = module.forward              # ← новый снапшот
```

#### `src/ctx_to_lora/modeling/lora_layer.py` (в `apply_random_repr`, обе ветки)

Перед навешиванием новой обёртки:
```python
if hasattr(module, "forward_lora_base"):
    module.forward = module.forward_lora_base
```

Идемпотентно: сколько бы раз `apply_random_repr` ни вызывался, он всегда
возвращает `module.forward` к patched‑базе и навешивает ровно одну свежую обёртку.

`forward_orig` (raw PEFT) остаётся как был — его использует `model.reset()` для
полного отката после `internalize`/`generate` в обычном workflow.

---

## Fix 5 — Детерминированный сэмплинг для apples‑to‑apples сравнения D2L vs base

### Симптом
В `eval_utils.py:893` использовался **глобальный** `np.random.permutation(len(ds))[:N]`
без явного seed. Каждый запуск (и каждый из двух скриптов
`d2l_skills.sh` / `base_model_skills.sh`) выбирал **разные** 100 индексов из
общих 534 → метрики D2L и base считались на **непересекающихся** подвыборках.

### Правка — `src/ctx_to_lora/eval_utils.py:887-908`

```python
eval_sample_seed = int(getattr(args, "seed", 42) or 42)
eval_rng = np.random.default_rng(eval_sample_seed)
...
val_indices = eval_rng.permutation(len(ds))[:max_eval_samples_per_ds]
```

- Локальный `default_rng` — глобальный numpy state не трогается.
- `args.seed` берётся из `cli_args.yaml` чекпоинта (для D2L = 42); если поля нет
  (base‑model'ный путь без чекпоинта) — fallback на 42.
- В обоих прогонах `len(ds)` = 534, seed = 42 → **одни и те же** 100 индексов.
- В лог пишется `seed=42`, чтобы было видно, что выборка детерминирована.

---

## Fix 6 — Base‑model eval оставлял датасет в 25 строк (без QA‑split)

### Симптом
```
... | 5/25 [00:35<02:22, 7.05s/it]
```
И это с явным `--max_val_samples_per_ds 100`. То есть «100» вообще не
сработало — потому что обрезать нечего: датасет всего 25.

### Корень
В `processing.py:295`:
```python
need_ctx_ids = ctx_model_max_len is not None
```
Весь блок чанкования + **`split_too_long_qas`** (`processing.py:453-522`) сидит
**внутри `if need_ctx_ids:`**. Для D2L `ctx_model_max_len = ctx_encoder.config.
max_position_embeddings` ставится в `eval_utils.py:750` (плюс LLMLingua/T2L/
GenerativeAdapter‑ветки тоже его ставят). А для **plain base‑model** пути
никто не ставил → `None` → `need_ctx_ids=False` → split не зовётся → датасет
остаётся 25 строк, каждая — multi‑turn диалог со всеми ~20 QA сразу.

В таком виде: (а) метрики не сравнимы с D2L; (б) `--max_val_samples_per_ds 100`
безмолвно игнорируется (selection of 100 of 25 returns 25); (в) генерация
делается всего по одной финальной реплике на огромный диалог.

### Правка — `src/ctx_to_lora/eval_utils.py`, после `if base_model is None`‑блока

```python
if ctx_model_max_len is None:
    ctx_model_max_len = base_model.config.max_position_embeddings
```

Эффект:
- `need_ctx_ids = True` для base‑model тоже → `processing.py:453+` отрабатывает,
  токенизирует контекст (тривиально, один чанк), и **`split_too_long_qas`
  разбивает по `max_qas_per_sample=1`** → 534 сэмпла.
- `--max_val_samples_per_ds 100` обрезает до 100 (тех же индексов, что и в
  D2L‑прогоне).
- При `--remove_context` контекст в чат не подаётся (`add_ctx_to_chat=False` в
  `eval_utils.py:845`); `ctx_ids` сидит в датасете, но не используется в forward
  модели → метрики правильные (closed‑book).

---

## Fix 7 — Raw HF.generate ругается на `ctx_*` kwargs

### Симптом
```
ValueError: The following `model_kwargs` are not used by the model:
['ctx_ids', 'ctx_attn_mask', 'n_ctx_chunks']
```

### Корень
Side‑эффект Fix 6: с `need_ctx_ids=True` для base‑model `generation_collator`
(`collator.py:106+`) кладёт в batch `ctx_ids`/`ctx_attn_mask`/`n_ctx_chunks`.
Trainer.predict вызывает `model.generate(**batch)`. Для D2L это
`ModulatedPretrainedModel.generate`, у которого `ctx_*` — явные параметры в
сигнатуре (`hypernet.py:1031-1042`), он их потребляет и в `base_model.generate`
не пробрасывает. Для **plain base‑model** `model = base_model = HF
Qwen3ForCausalLM`, у которого таких kwargs нет → `ValueError`.

### Правка — `src/ctx_to_lora/eval_utils.py`, сразу после `get_model(...)`

Сначала `add_tracker` (он требует bound‑method и сам подменяет
`base_model.generate` своей tracker‑closure'ой), потом наш strip‑wrapper:

```python
add_tracker(base_model.generate, "generate")          # 1) tracker над bound generate
_tracked_base_generate = base_model.generate          # 2) уже tracker'd версия
def _base_generate_strip_ctx(*args, **kwargs):
    for _k in ("ctx_ids", "ctx_attn_mask", "ctx_position_ids", "n_ctx_chunks"):
        kwargs.pop(_k, None)
    return _tracked_base_generate(*args, **kwargs)
base_model.generate = _base_generate_strip_ctx        # 3) внешняя обёртка стрипера
```

**Порядок важен**: если сначала навесить strip‑wrapper, а потом `add_tracker`
— получим `ValueError: add_tracker expects a bound method` (плоская функция не
имеет `__self__`). Сначала tracker (он работает на bound‑method), потом строгий
wrapper.

Цепочка вызова на eval:
```
Trainer.predict → strip_wrapper → tracker_closure → original HF Qwen3.generate
```
- `ctx_*` отбрасываются strip‑wrapper'ом до того, как HF их увидит.
- Tracker корректно меряет время/память реального HF generate.
- D2L путь не задет: его `ModulatedPretrainedModel.generate` всё равно потребляет
  `ctx_*` сам, до `self.base_model.generate(...)`.

---

## Анатомия слоёв обёртки `down_proj.forward`

После всех правок:

```
peft.lora.Linear.forward                           ← forward_orig (raw)
        │
        ▼
partial(random_repr_forward,                       ← forward_lora_base (snapshot)
        self=module,
        lora_dropout_p, scaling)
        │
        ▼  (apply_random_repr, packed)             ← train (sequence packing)
partial(forward_lora_base,
        n_queries, tot_q, seq_lens, tot_len,
        coeffs, n_ctx_chunks, repr_seeds, generator)
        │
        ▼  (apply_random_repr, unpacked)           ← eval / HF.generate
make_wrapper(partial(forward_lora_base,
                     n_queries, tot_q, coeffs,
                     n_ctx_chunks, repr_seeds, generator))
   └─→ fn(x, …): добавляет seq_lens=[seq_len]*tot_q,
                          tot_len=seq_len*tot_q  на каждом вызове
```

`make_wrapper` нужен только в unpacked‑режиме, чтобы `seq_lens`/`tot_len`
пересчитывались между шагами generate (kv‑cache меняет `x.size(1)`).

---

## Анатомия `base_model.generate` для base‑model eval

После Fix 7:
```
HF Qwen3ForCausalLM.generate (bound)
        │
        ▼  add_tracker  →  setattr(base_model, "generate", tracked)
tracker_closure
        │
        ▼  monkey‑patch  →  base_model.generate = _base_generate_strip_ctx
strip_wrapper
   └─→ pop ctx_ids / ctx_attn_mask / ctx_position_ids / n_ctx_chunks
   └─→ call tracker_closure → original HF generate
```

---

## Что фикс не покрывает

1. **`--eval_batch_size_gen > 1`** — `random_repr_forward` написан под
   `[1, tot_len, d_in]`. При батче >1 нужна ещё одна ветка / переписать einsum'ы
   с явным `bs`. Сейчас держать batch=1.
2. **`--use_iterative_mode`** для random‑чекпоинта — no‑op. Ветка в
   `hypernet.py:468` требует `aggregator.layer_to_layer == True`, что эквивалентно
   `ctx_encoder_type == per_layer_activations`. У нас `early_exit` →
   `layer_to_layer=False`, флаг тихо игнорируется. Iterative имеет смысл только
   для не‑random чекпоинта (`qwen_diverse_sft_norandom.sh`).
3. **`max_ctx_chunk_len`** — лучше держать ≤ `max_packed_ctx_len` из тренировки
   (в нашем случае 6144) для согласованности геометрии.
4. **Stratified sampling по контекстам** — Fix 5 даёт детерминированную выборку
   100 из 534, но это случайные `(ctx, prompt, response)` пары; покрытие всех
   25 документов не гарантировано. Если важно — нужен отдельный код, который
   группирует по `id` контекста и берёт N штук с каждого. Сейчас не сделано.
5. **Лишняя токенизация контекста для base‑model'а** (Fix 6): `ctx_ids` сидит
   в датасете, занимает память, но при `--remove_context` не используется.
   Накладные расходы — секунды на старте, не критично.

---

## Чек‑лист для запуска random‑repr eval

### D2L (`d2l_skills.sh`)
```bash
export D2L_USE_RANDOM_REPR=1                # обязательно
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled uv run run_eval.py \
    --checkpoint_path train_outputs/runs/<RUN>/<CKPT>/pytorch_model.bin \
    --datasets diverse_sft --split validation \
    --max_ctx_chunk_len 6144 \
    --eval_batch_size_gen 1 \                # обязательно
    --max_val_samples_per_ds 100             # детерминирован seed=42
```

### Base‑model (`base_model_skills.sh`)
```bash
# с контекстом (full‑ICL)
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled uv run run_eval.py \
    --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
    --datasets diverse_sft --split validation \
    --eval_batch_size_gen 1 --max_new_tokens 512 \
    --output_dir eval_results/qwen3_4b_skills_full \
    --max_val_samples_per_ds 100

# closed‑book (без контекста)
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled uv run run_eval.py \
    --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
    --datasets diverse_sft --split validation \
    --eval_batch_size_gen 1 --remove_context --max_new_tokens 512 \
    --output_dir eval_results/qwen3_4b_skills_no_ctx \
    --max_val_samples_per_ds 100
```

Все три прогона теперь оценивают **одни и те же 100 сэмплов** (Fix 5),
геометрия датасета одинаковая (Fix 6), и raw HF generate не падает на
`ctx_*` kwargs (Fix 7).

Метрика для `diverse_sft` — **mean rougeL F1** (рассчитывается через
`rouge_score.RougeScorer(["rougeL"], use_stemmer=True)`, `compute_rouge` в
`metrics.py:34-44`), один эталон на сэмпл (`responses[i]`), greedy decode
(`do_sample=False`), плюс length‑binned разбивка `rougeL.f1_len_*`.
