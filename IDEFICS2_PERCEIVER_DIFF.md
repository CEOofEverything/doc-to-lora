# Idefics2Perceiver: схемы и код

## Исходный Forward Pass

```text
Активации текста (features) из ctx_encoder

       ↓

[bs, seq_len, input_size]
сырые данные из базовой LLM

       ↓

self.modality_projection
(trainable MLP)
подгоняет размерность под внутренний стандарт Perceiver

       ↓

[bs, seq_len, hidden_size]
спроецированный текст

       ↓

self.encoder
(trainable, Cross-Attention)
обучаемые latents_q смотрят на текст и впитывают смысл

       ↓

[bs, n_latents, hidden_size]
длинный текст сжат в короткий набор векторов

       ↓

self.decoder
(trainable, Cross-Attention + Self-Attention)
финальная обработка сжатых векторов между собой

       ↓

[bs, n_latents, hidden_size]
финальный выход агрегатора

       ↓

Уходит в ResMLPBlock и Head для генерации матриц LoRA
```

## Модифицированный Forward Pass с чанкингом

```text
Входные данные: features, n_ctx_chunks (число чанков), repr_seeds (сиды)
       ↓
  [bs, seq_len, input_size]
       ↓
====================================================
ВЕТВЛЕНИЕ ЛОГИКИ (если use_random_repr = True)
====================================================

ВЕТКА А (Текстовая)
(Обучается, requires_grad=True)

features
       ↓
self.modality_projection
       ↓
[bs, seq_len, hidden_size]
       ↓
self.encoder
       ↓
[bs, n_latents, hidden_size]
       ↓
ctx_latents
(Смысл текста)

====================================================

ВЕТКА Б (Базис для чанков)
(НЕ обучается, torch.no_grad())

Цикл по документам (n_chunks, seed):
       ↓
generator.manual_seed(seed)
       ↓
генерация repr_A: [n_chunks, n_reprs, r, out_features]
генерация repr_B: [n_chunks, n_reprs, in_features, r]
       ↓
self.modality_projection(repr_A)
       ↓
[n_chunks, n_reprs, r, hidden_size] — projected_A
       ↓
torch.matmul(repr_B, projected_A)
(ОГРОМНАЯ матрица)
       ↓
[n_chunks, n_reprs, in_features, hidden_size]
       ↓
view(1, -1, hidden_size)
сплющивание в "длинный текст" (73728 токенов)
       ↓
[1, n_chunks * n_reprs * in_features, hidden_size]
       ↓
self.encoder([1, n_chunks * n_reprs * in_features, hidden_size] "фейковый текст")
       ↓
[n_chunks, n_latents, hidden_size]
       ↓
Сборка выходов цикла в один тензор батча
       ↓
repr_latents
(Белый шум, зависящий от сида)

====================================================
СЛИЯНИЕ ВЕТОК
====================================================

       ↓
  latents = ctx_latents + repr_latents 
  (К осмысленному вектору просто прибавляется шум документа)
       ↓
  [bs, n_latents, hidden_size]
       ↓
  self.decoder                     (trainable)
       ↓
  [bs, n_latents, hidden_size]     — финальный выход агрегатора
       ↓
  Уходит в Head предсказывать КОЭФФИЦИЕНТЫ
```

## Сам код

```python
class Idefics2Perceiver(Idefics2PreTrainedModel):
    def __init__(
        self,
        encoder_config: Idefics2PerceiverConfig,
        decoder_config: Idefics2PerceiverConfig,
        use_random_repr: bool = False,  # added  # НОВОЕ
    ):
        super().__init__(encoder_config)
        self.modality_projection = Idefics2MLP(
            hidden_size=encoder_config.input_size,  # 2304?
            intermediate_size=(
                encoder_config.intermediate_size_factor * encoder_config.input_size
            ),
            output_size=encoder_config.hidden_size,
            hidden_act=encoder_config.hidden_act,
        )
        self.encoder = Idefics2PerceiverResampler._from_config(encoder_config)
        self.decoder = Idefics2PerceiverResampler._from_config(decoder_config)

        self.use_random_repr = use_random_repr  # НОВОЕ
        self.n_reprs = 8  # probably change  # НОВОЕ
        self.n_layers = 26  # for gemma-2-2b-it  # НОВОЕ
        self.r = 8  # НОВОЕ
        self.in_features = 9216  # for gemma-2-2b-it  # НОВОЕ
        self.out_features = 2304  # for gemma-2-2b-it  # НОВОЕ

        logger.debug(
            f"Using n_repr = {self.n_reprs} with d_in = {self.in_features}, "
            f"d_out = {self.out_features}"
        )  # НОВОЕ

    def forward(
        self,
        features: torch.Tensor,  # Float[Tensor, "bs seq_len feature_dim"]
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        n_ctx_chunks: torch.LongTensor | None = None,  # НОВОЕ
        repr_seeds: torch.LongTensor | None = None,  # НОВОЕ
        generator: torch.Generator | None = None,  # НОВОЕ
    ):
        bs = (
            features.size(0) if position_ids is None else
            (position_ids == 0).sum()
        )
        # print(f"Starting with {features.size() = } and bs = {bs.item()}", flush=True)

        projected_inputs = self.modality_projection(features)

        ctx_latents = self.encoder(  # [bs, n_latents, dim]
            context=projected_inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            precalculated_bs=bs,  # for speedup?  # НОВОЕ
        )
        # print(f"In between with {ctx_latents.size() = }", flush=True)

        if self.use_random_repr:  # НОВОЕ
            repr_latents = []  # НОВОЕ
            # for layer_idx in range(self.n_layers):  # mb replace with layer_indices?
                # stat = torch.cuda.memory.memory_allocated(device=features.device)
                # print(f"Allocated for {layer_idx = }: {stat / (1024 ** 3):.1f}Gb", flush=True)

            for n_chunks, seed in zip(n_ctx_chunks, repr_seeds):  # НОВОЕ
                # seed = seed.item()
                # generator.manual_seed(seed + layer_idx)
                generator.manual_seed(seed.item())  # НОВОЕ
                # stat = torch.cuda.memory.memory_allocated(device=features.device)
                # print(f"Allocated for seed = {seed.item()}: {stat / (1024 ** 3):.1f}Gb", flush=True)

                with torch.no_grad():  # НОВОЕ
                    repr_A = torch.randn(  # НОВОЕ
                        (n_chunks, self.n_reprs, self.r, self.out_features),
                        generator=generator,
                        device=features.device,
                        dtype=features.dtype,
                    )
                    repr_B = torch.randn(  # НОВОЕ
                        (n_chunks, self.n_reprs, self.in_features, self.r),
                        generator=generator,
                        device=features.device,
                        dtype=features.dtype,
                    )

                    projected_repr_A = self.modality_projection(repr_A)  # НОВОЕ
                    random_repr = torch.matmul(repr_B, projected_repr_A).view(  # НОВОЕ
                        1, -1, self.modality_projection.down_proj.out_features
                    )

                    repr_position_ids = torch.arange(  # НОВОЕ
                        self.n_reprs * self.in_features, device=features.device
                    ).unsqueeze(0)
                    repr_position_ids = torch.tile(  # НОВОЕ
                        repr_position_ids, (1, n_chunks)
                    )

                    chunk_repr_latents = self.encoder(  # НОВОЕ
                        context=random_repr,
                        position_ids=repr_position_ids,
                        precalculated_bs=n_chunks,
                    )
                    repr_latents.append(chunk_repr_latents)  # НОВОЕ

            repr_latents = torch.cat(repr_latents, dim=0)  # НОВОЕ
            # print(f"Now {repr_latents.size() = }", flush=True)

            latents = ctx_latents + repr_latents  # НОВОЕ
        else:
            latents = ctx_latents

        latent_position_ids = torch.arange(
            self.encoder.n_latents, device=features.device
        ).unsqueeze(0)
        latent_position_ids = torch.tile(latent_position_ids, (1, bs))

        outputs = self.decoder(
            context=latents,
            position_ids=latent_position_ids,
            precalculated_bs=bs,  # НОВОЕ
        )
        # print(f"Finishing with {outputs.size() = }", flush=True)

        # before: return outputs
        return outputs
```
