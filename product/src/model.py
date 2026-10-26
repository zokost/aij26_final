"""Модель сегментации + проверка регламентных ограничений (FLOPs и латентность).

Регламент:
  * <= 100 GFLOPs на изображение в СТРОГИХ FLOPs (1 MAC = 2 FLOPs);
  * <= 50 мс на изображение на H100.

Считаем строгие FLOPs через ``torch.utils.flop_counter.FlopCounterMode`` — именно
его регламент называет арбитром (fvcore/thop/ptflops возвращают MACs, их надо
удваивать).
"""
from __future__ import annotations

import copy
import time

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

# Энкодеры с ФИКСИРОВАННЫМ входом: Swin, MaxViT, CoAtNet строят таблицы
# относительных позиций и разбиение на окна в конструкторе, поэтому им надо
# передать img_size при создании, иначе они падают на несовпадении с дефолтными
# 224. Для нас это не ограничение: squish подаёт фиксированный квадрат.
# ВАЖНО: smp.create_model глотает лишние kwargs, а классы архитектур передают их
# в get_encoder -> timm.create_model. Поэтому строим только через класс.
FIXED_INPUT = ("swin", "maxvit", "coatnet", "davit", "hiera")

ARCHS = {
    "unet": smp.Unet,
    "manet": smp.MAnet,
    "pan": smp.PAN,
    "pspnet": smp.PSPNet,
    "unetplusplus": smp.UnetPlusPlus,
    "fpn": smp.FPN,
    "deeplabv3plus": smp.DeepLabV3Plus,
    "segformer": smp.Segformer,
    "upernet": smp.UPerNet,
}


class DeepSupervisionNet(nn.Module):
    """Two spatial training heads; inference uses the unchanged backbone.

    Heads read the stride-8 and stride-16 MiT features. Their logits are
    enlarged to the full target resolution, so small GT regions are never
    erased by downsampling the mask. Call through ``forward`` (including
    under DDP) to keep gradient synchronization working for every head.
    """

    def __init__(self, backbone: nn.Module, width: int = 64):
        super().__init__()
        self.backbone = backbone
        self.feature_indices = (-3, -2)
        channels = backbone.encoder.out_channels
        self.auxiliary_heads = nn.ModuleList()
        for index in self.feature_indices:
            ch = channels[index]
            if ch <= 0:
                raise ValueError("deep supervision requires two spatial encoder stages")
            head = nn.Sequential(
                nn.Conv2d(ch, width, 3, padding=1, bias=False),
                nn.GroupNorm(8, width),
                nn.ReLU(inplace=True),
                nn.Conv2d(width, 1, 1),
            )
            # New heads cannot perturb the pretrained encoder on their first
            # backward pass. Main segmentation is unchanged at initialization.
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            self.auxiliary_heads.append(head)

    def forward(self, x: torch.Tensor, return_auxiliary: bool = False):
        if not return_auxiliary:
            return self.backbone(x)
        features = self.backbone.encoder(x)
        logits = self.backbone.segmentation_head(self.backbone.decoder(features))
        auxiliary = [
            F.interpolate(head(features[index]), size=x.shape[-2:],
                          mode="bilinear", align_corners=False)
            for index, head in zip(self.feature_indices, self.auxiliary_heads)
        ]
        return logits, auxiliary

    def load_base_state_dict(self, state_dict: dict) -> None:
        self.backbone.load_state_dict(state_dict)


class GlobalDetectionNet(nn.Module):
    """Сегментатор + голова «есть ли на кадре манипуляция», только для обучения.

    Голова читает последнюю карту энкодера и на инференсе не вызывается:
    ``forward`` отдаёт ровно те же логиты, что и базовая модель, поэтому FLOPs
    посылки не меняются ни на грамм. Обучение идёт с
    ``criterion(маска) + w * BCE(есть ли манипуляция)``.

    Идея и вес 0.2 — из ``danzl_branch``. Там на базе `P2_effb3_896_e40plus`
    она дала **+0.0083 на лидерборде** (0.9097 -> 0.9180) при +0.0034 на
    валидации, то есть перенос оказался вдвое сильнее валидационного замера.
    Здесь обобщено с Unet на любой smp-сегментатор (нам нужен FPN).
    """

    def __init__(self, base: nn.Module):
        super().__init__()
        self.backbone = base
        head = nn.Linear(base.encoder.out_channels[-1], 1)
        # Голову инициализируем нулями: на первом шаге её выход и градиент в
        # энкодер строго нулевые, поэтому она ничего не ломает, а дальше
        # раскачивается сама. У реализации из danzl_branch голова стартует
        # случайной, и на дообучении это стоило −0.0189 AIC на первой эпохе —
        # при коротком прогоне половина бюджета уходит на восстановление.
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        self.global_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            head,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward_with_detection(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone.encoder(x)
        logits = self.backbone.segmentation_head(self.backbone.decoder(features))
        return logits, self.global_head(features[-1]).squeeze(1)

    def load_base_state_dict(self, state_dict: dict) -> None:
        """Загружает веса обычного прогона: голова остаётся свежеинициализированной."""
        self.backbone.load_state_dict(state_dict)


class GeneratorClsNet(nn.Module):
    """Сегментатор + голова «каким генератором сделан кадр», только для обучения.

    Зачем. Шесть публичных forensic-моделей на нашей валидации дают Dice
    0.16-0.38 против наших 0.92, и все обучены на сплайсинге, тогда как у нас
    100% инпейнтинг. Знание «чем один генеративный процесс отличается от
    другого» импортировать неоткуда — зато метка генератора лежит прямо в имени
    файла, то есть его можно выучить на своих же данных, без внешних картинок.

    Устройство повторяет ``GlobalDetectionNet``: голова читает последнюю карту
    энкодера, на инференсе не вызывается, поэтому FLOPs посылки не меняются.
    Инициализация нулями — на первом шаге выход и градиент в энкодер строго
    нулевые, голова ничего не ломает и раскачивается сама (у случайной
    инициализации это стоило -0.0189 AIC на первой эпохе).
    """

    def __init__(self, base: nn.Module, num_classes: int):
        super().__init__()
        self.backbone = base
        head = nn.Linear(base.encoder.out_channels[-1], num_classes)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        self.gen_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            head,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward_with_detection(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone.encoder(x)
        logits = self.backbone.segmentation_head(self.backbone.decoder(features))
        return logits, self.gen_head(features[-1])

    def load_base_state_dict(self, state_dict: dict) -> None:
        self.backbone.load_state_dict(state_dict)


# Три классических высокочастотных ядра SRM (Fridrich & Kodovsky), те же, что
# используют RGB-N и MVSS-Net. Ядра с нулевой суммой: они убивают содержимое
# кадра и оставляют шумовой остаток, в котором и живут следы вставки.
SRM_KERNELS = [
    [[0, 0, 0, 0, 0],
     [0, -1, 2, -1, 0],
     [0, 2, -4, 2, 0],
     [0, -1, 2, -1, 0],
     [0, 0, 0, 0, 0]],
    [[-1, 2, -2, 2, -1],
     [2, -6, 8, -6, 2],
     [-2, 8, -12, 8, -2],
     [2, -6, 8, -6, 2],
     [-1, 2, -2, 2, -1]],
    [[0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 1, -2, 1, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0]],
]
SRM_SCALE = [4.0, 12.0, 2.0]      # нормировка ядер, как в оригинале


class SRMStem(nn.Module):
    """Фиксированные SRM-остатки, приклеенные к RGB как дополнительные каналы.

    Зачем: наши модели видят только RGB и хорошо ловят диффузионную перерисовку,
    но проваливаются на числовом семействе (Dice 0.647 против 0.875) — а правка
    цифр оставляет след не в семантике, а в шуме и артефактах сжатия. В
    форензике первый стандартный ход — дать сети высокочастотный поток; здесь он
    стоит один фиксированный свёрточный слой и не обучается.

    Каналов ровно три (по одному на ядро, применяется к сумме RGB), а не девять:
    девять дали бы 101.0 GFLOPs при лимите 100. С тремя выходит 99.40 — запас
    0.60 G, вдвое меньше нашего обычного, но в регламент укладывается.

    Добивка REPLICATE, а не нулями: ядра нулевой суммы гасят ровную заливку до
    5e-8 внутри кадра, но у нулевой добивки на рамке остаётся всплеск 0.35 —
    фальшивый «след подделки» по краю каждого кадра, за который модель может
    зацепиться как за ярлык.
    """

    def __init__(self, normalize: bool = False):
        super().__init__()
        w = torch.tensor(SRM_KERNELS, dtype=torch.float32)                  # (3,5,5)
        w = w / torch.tensor(SRM_SCALE).view(3, 1, 1)
        # (3,3,5,5): каждое ядро применяется ко всем каналам и суммируется
        self.register_buffer("weight", w.unsqueeze(1).repeat(1, 3, 1, 1) / 3.0)
        # Замер на реальных данных: у нормализованного RGB ст. отклонение 1.345,
        # у остатков 0.0887 — разница в 15 раз. Предобученный стем получает
        # каналы несопоставимого масштаба и первые эпохи тратит на адаптацию
        # вместо пользы: в прогоне без нормировки отставание от контроля на
        # первой эпохе −0.0705, и только к пятой оно сжалось до −0.0071.
        # BatchNorm без affine выравнивает масштаб, не добавляя обучаемых
        # параметров. ПРОВЕРЕНО И ОТКЛЮЧЕНО: вариант с нормировкой закончил 8
        # эпох на 0.9050 против 0.9119 без неё. Диагноз оказался неверен —
        # выравнивание не убрало стартовую яму (−0.0777 на первой эпохе против
        # −0.0705), зато стёрло АБСОЛЮТНУЮ величину шумового остатка, а это и
        # есть форензический признак: у вставленной области энергия шума иная,
        # чем у исходной. Стандартизация по батчу убивает ровно тот сигнал,
        # ради которого фильтры и добавлялись.
        self.norm = nn.BatchNorm2d(3, affine=False) if normalize else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = F.conv2d(F.pad(x, (2, 2, 2, 2), mode="replicate"), self.weight)
        return torch.cat([x, self.norm(res)], dim=1)


class SRMNet(nn.Module):
    """Сегментатор поверх RGB + SRM-остатков."""

    # Внутренний backbone шестиканальный, но внешний вход модели остаётся RGB:
    # ещё три канала вычисляет SRMStem. Счётчики не должны угадывать внешний
    # интерфейс по первой зарегистрированной nn.Conv2d внутри backbone.
    input_channels = 3

    def __init__(self, base: nn.Module):
        super().__init__()
        self.stem = SRMStem()
        self.backbone = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.stem(x))

    def load_base_state_dict(self, state_dict: dict) -> None:
        """Load an RGB checkpoint without changing its initial function.

        The SRM backbone has a six-channel first convolution: RGB followed by
        three fixed residual channels.  Copy the trained RGB weights verbatim
        and initialise only the new channel weights with zeros.  Consequently
        the wrapped model produces exactly the source model's logits before
        fine-tuning, while gradients can still teach the new SRM weights.
        """
        target = self.backbone.state_dict()
        adapted = dict(state_dict)
        mismatches = []
        for name, source_value in state_dict.items():
            if name not in target or source_value.shape == target[name].shape:
                continue
            target_value = target[name]
            can_expand_rgb = (
                source_value.ndim == 4
                and source_value.shape[1] == 3
                and target_value.shape[1] == 6
                and source_value.shape[0] == target_value.shape[0]
                and source_value.shape[2:] == target_value.shape[2:]
            )
            if not can_expand_rgb:
                mismatches.append(
                    f"{name}: source {tuple(source_value.shape)} -> "
                    f"target {tuple(target_value.shape)}"
                )
                continue
            expanded = torch.zeros_like(target_value)
            expanded[:, :3].copy_(source_value)
            adapted[name] = expanded
        if mismatches:
            raise RuntimeError("SRM checkpoint shape mismatch: " + "; ".join(mismatches))
        self.backbone.load_state_dict(adapted)


class RefineNet(nn.Module):
    """Уточнение границ: восстанавливать разрешение свёртками, а не интерполяцией.

    Зачем. Разбор ошибок лидера: 63.4% всей потери Dice — это кадры, где площадь
    предсказана верно, а границы неточны. Полных промахов лишь 2%. Декодер
    SegFormer выдаёт карту на 1/4 разрешения (208x208 при входе 832) и растягивает
    её билинейно в 4 раза — граница размывается именно там.

    Здесь вместо одного растяжения идут два шага с обучаемой свёрткой между ними.
    Стоит доли гигафлопса: свёртки работают на 16 каналах, а не на сотнях.
    """

    def __init__(self, base: nn.Module, width: int = 8, in_channels: int | None = None):
        super().__init__()
        self.backbone = base
        # Ширину выхода декодера НЕЛЬЗЯ угадывать. У Segformer в smp декодер не
        # выставляет out_channels, и прежний фолбэк на 256 ронял сборку с
        # --decoder-channels 128: «expected input[1, 128, 208, 208] to have 256
        # channels». Поэтому build_model передаёт её явно, а угадывание
        # осталось только для вызовов без этого аргумента.
        ch = in_channels or getattr(base.decoder, "out_channels", None) or 256
        self.refine = nn.Sequential(
            # 1x1 на входе: снижение канальности с 256 до 16 не требует
            # пространственного ядра, а 3x3 здесь стоило бы 3.2 GFLOPs из 4.2 —
            # именно оно выбивало модуль за бюджет (102.75 G при лимите 100)
            nn.Conv2d(ch, width, 1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(width, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.encoder(x)
        dec = self.backbone.decoder(feats)
        out = self.refine(dec)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out

    def load_base_state_dict(self, state_dict: dict) -> None:
        """Load every compatible EXT63 backbone weight; refine stays new."""
        self.backbone.load_state_dict(state_dict)


def build_model(arch: str = "unet",
                encoder: str = "resnet34",
                encoder_weights: str | None = "imagenet",
                img_size: int = 512,
                global_det: bool = False,
                srm: bool = False,
                refine: bool = False,
                gen_cls: int = 0,
                extra_channels: int = 0,
                decoder_channels: int | None = None) -> nn.Module:
    """Собирает сегментатор с одним выходным каналом (логиты).

    ``arch="mesorch_p"`` — внешняя архитектура (AAAI 2025), подключена для
    честного сравнения: тот же лосс, тот же сплит, та же проверка лимитов.
    Энкодер для неё не задаётся (внутри фиксированы ConvNeXt-T + MiT-B3), а
    разрешение важно, потому что финальный upsample у неё зашит в конструктор.
    """
    if arch == "mesorch_p":
        from mesorch_wrap import MesorchP
        return MesorchP(img_size=img_size, pretrained=encoder_weights is not None)
    if arch == "cuenet":
        if encoder != "convnextv2_large":
            raise ValueError("published Cue-Net weights require convnextv2_large")
        if global_det or srm or refine or gen_cls or extra_channels or decoder_channels is not None:
            raise ValueError("Cue-Net uses its original segmentation decoder without extra branches")
        from cuenet_pretrained_adapter import build_cuenet
        return build_cuenet(pretrained=encoder_weights is not None)
    if arch == "trufor_budget":
        if global_det or srm or refine or gen_cls or extra_channels:
            raise ValueError("budget TruFor includes its original two streams without extra branches")
        from trufor_budget_adapter import build_budget_trufor
        return build_budget_trufor(variant=encoder, decoder_channels=decoder_channels or 128,
                                  pretrained=encoder_weights is not None)
    if arch not in ARCHS:
        raise ValueError(f"неизвестная архитектура {arch!r}, доступны: {sorted(ARCHS)}")
    extra = {}
    if any(k in encoder for k in FIXED_INPUT):
        h, w = _hw(img_size)
        if h != w:
            raise SystemExit(f"{encoder} требует квадратный вход, получено {h}x{w}")
        extra["img_size"] = h
    if decoder_channels is not None:
        # ИМЯ ПАРАМЕТРА ЗАВИСИТ ОТ АРХИТЕКТУРЫ. У Segformer в smp он называется
        # decoder_segmentation_channels; передача decoder_channels проваливалась
        # в **kwargs и МОЛЧА игнорировалась — замер 08.09 показал одинаковые
        # 96.01 G и 24.72 M параметров при значениях 64/96/128 и без флага.
        import inspect as _inspect
        _params = _inspect.signature(ARCHS[arch].__init__).parameters
        _name = ("decoder_segmentation_channels"
                 if "decoder_segmentation_channels" in _params else "decoder_channels")
        if _name not in _params:
            raise SystemExit(f"{arch} не принимает ширину декодера")
        extra[_name] = decoder_channels
    if False:
        # У UperNet декодер стоит дороже энкодера: с ConvNeXt-Tiny при 832 это
        # 403 G из которых на энкодер приходится меньше четверти. Сузив декодер
        # с 256 до 96 каналов, ту же схему (PPM + FPN), которая дала 0.9619 на
        # ЛБ вне бюджета, удаётся уложить в 96.9 G при 640.
        extra["decoder_channels"] = decoder_channels
    base = ARCHS[arch](
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=(6 if srm else 3) + extra_channels,
        classes=1,
        **extra,
    )
    if refine:
        return RefineNet(base, in_channels=decoder_channels)
    if srm:
        # smp сам растягивает веса первой свёртки под 6 каналов, предобучение
        # энкодера при этом сохраняется
        return SRMNet(base)
    if gen_cls:
        return GeneratorClsNet(base, gen_cls)
    return GlobalDetectionNet(base) if global_det else base


def _hw(img_size: int | tuple[int, int]) -> tuple[int, int]:
    """Вход бывает квадратным (832) и прямоугольным ((672, 1024))."""
    return img_size if isinstance(img_size, (tuple, list)) else (img_size, img_size)


@torch.no_grad()
def _model_input_channels(model: nn.Module, default: int = 3) -> int:
    """Сколько каналов ждёт внешний ``forward`` модели."""
    declared = getattr(model, "input_channels", None)
    if declared is not None:
        return int(declared)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            return m.in_channels
    return default


def count_flops(model: nn.Module, img_size: int | tuple[int, int] = 512,
                device: str = "cuda", in_channels: int | None = None) -> float:
    """Строгие FLOPs на ОДНО изображение (batch=1).

    ``in_channels`` позволяет явно задать внешний интерфейс. Обычно он берётся
    из ``model.input_channels`` или первой свёртки; для SRM эти числа различны,
    потому что фиксированный stem сам строит дополнительные каналы из RGB.
    """
    model = model.to(device).eval()
    h, w = _hw(img_size)
    if in_channels is None:
        in_channels = _model_input_channels(model)
    x = torch.randn(1, in_channels, h, w, device=device)
    counter = FlopCounterMode(display=False)
    with counter:
        model(x)
    return counter.get_total_flops()


@torch.no_grad()
def benchmark_latency(model: nn.Module,
                      img_size: int | tuple[int, int] = 512,
                      batch_size: int = 16,
                      device: str = "cuda",
                      dtype: torch.dtype = torch.float16,
                      n_warmup: int = 5,
                      n_iter: int = 20) -> float:
    """Среднее время на изображение, мс. Меряем батчем — так гоняют инференс.

    Работаем на КОПИИ: вызывающий код (обучение) ожидает свою модель нетронутой.

    Смешанная точность включается через ``autocast``, а не переводом весов в
    half: часть архитектур (например, Mesorch-P) намеренно считает softmax
    внимания в float32, и жёсткий ``.half()`` на них падает. Autocast к тому же
    ближе к тому, как инференс гоняют на практике.
    """
    model = copy.deepcopy(model).to(device).eval()
    h, w = _hw(img_size)
    x = torch.randn(batch_size, _model_input_channels(model), h, w, device=device)

    with torch.autocast("cuda", dtype=dtype):
        for _ in range(n_warmup):
            model(x)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_iter):
            model(x)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed / (n_iter * batch_size) * 1000


def check_limits(model: nn.Module, img_size: int | tuple[int, int],
                 batch_size: int = 16,
                 flops_limit: float = 100e9, latency_limit_ms: float = 50.0) -> dict:
    """Прогоняет обе проверки и печатает вердикт."""
    flops = count_flops(model, img_size)
    latency = benchmark_latency(model, img_size, batch_size=batch_size)
    ok_flops = flops <= flops_limit
    ok_lat = latency <= latency_limit_ms
    h, w = _hw(img_size)
    print(f"вход {h}x{w}")
    print(f"FLOPs/изобр.: {flops / 1e9:7.2f} G (лимит {flops_limit / 1e9:.0f} G) "
          f"-> {'OK' if ok_flops else 'ПРЕВЫШЕН'}")
    print(f"Латентность:  {latency:7.2f} мс (лимит {latency_limit_ms:.0f} мс) "
          f"-> {'OK' if ok_lat else 'ПРЕВЫШЕНА'}")
    return {"gflops": flops / 1e9, "latency_ms": latency, "ok": ok_flops and ok_lat}
