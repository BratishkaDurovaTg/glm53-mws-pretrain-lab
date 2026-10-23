"""Sentence-frame banks. Frames embed {T}=mention token, {C}=component word.

Frames assert the mention's latent status in their wording — that wording is what makes
gold derivable. Noise lines contain no technical literals.
"""

import random

from compactex.schemas import LiteralKind

# Generic frames usable for any kind (kind-neutral assertive wording)
GENERIC: dict[str, dict[str, list[str]]] = {
    "current": {
        "ru": [
            "проблема связана с {T}",
            "сейчас воспроизводится {T}",
            "активная проблема: {T}",
            "клиенты видят {T}",
        ],
        "en": [
            "the active issue involves {T}",
            "we are currently seeing {T}",
            "right now the failing bit is {T}",
        ],
    },
    "historical": {
        "ru": [
            "ещё вчера наблюдалось {T}",
            "раньше мы ловили {T}",
            "до вчерашнего дня было {T}",
            "на прошлой неделе отмечалось {T}",
        ],
        "en": [
            "yesterday we observed {T}",
            "previously we hit {T}",
            "last week there was {T}",
        ],
    },
    "resolved": {
        "ru": [
            "{T} уже исправлено после перезапуска",
            "раньше было {T}, но после фикса всё чисто",
        ],
        "en": [
            "{T} was fixed after the restart",
            "we used to see {T}, gone after the fix",
        ],
    },
    "hypothetical": {
        "ru": [
            "похоже, дело в {T}",
            "возможно, проблема в {T}",
            "подозреваю {T}",
            "кажется, это {T}",
        ],
        "en": [
            "this might be {T}",
            "perhaps the cause is {T}",
            "I suspect {T}",
        ],
    },
    "mentioned_only": {
        "ru": [
            "это точно не {T}",
            "в документации упоминается {T}",
            "если вдруг увидите {T} — напишите",
            "мы предполагали {T}, но подтверждения нет",
            "в соседнем проекте видели {T}, у нас такого нет",
        ],
        "en": [
            "it is definitely not {T}",
            "the docs mention {T}",
            "if you ever see {T}, let us know",
        ],
    },
}

# Kind-specific flavour (overrides generic for realism)
KIND_FRAMES: dict[tuple[str, str], dict[str, list[str]]] = {
    ("error_code", "current"): {
        "ru": [
            "сервис {C} возвращает {T}",
            "{C} постоянно отвечает {T}",
            "при оплате падает с {T}",
            "шлюз отдаёт {T}",
        ],
        "en": [
            "the {C} service returns {T}",
            "{C} keeps responding with {T}",
            "payments fail with {T}",
        ],
    },
    ("error_code", "historical"): {
        "ru": ["вчера {C} отдавал {T}", "до обновления ловили {T} от {C}"],
        "en": ["yesterday {C} returned {T}", "before the update {C} threw {T}"],
    },
    ("error_code", "resolved"): {
        "ru": ["{C} уже поправили — раньше это был {T}"],
        "en": ["{C} is fixed now — it used to be {T}"],
    },
    ("error_code", "hypothetical"): {
        "ru": ["похоже на {T} от {C}", "думаем, это {T}"],
        "en": ["looks like {T} from {C}", "we think it is {T}"],
    },
    ("error_code", "mentioned_only"): {
        "ru": ["это не {T}", "в changelog пишут про {T}", "не путайте с {T}"],
        "en": ["it is not {T}", "the changelog mentions {T}", "do not confuse it with {T}"],
    },
    ("version", "current"): {
        "ru": ["кластер на версии {T}", "стоит {T}", "обновились до {T}"],
        "en": ["the cluster runs {T}", "we are on {T}", "upgraded to {T}"],
    },
    ("version", "historical"): {
        "ru": ["до этого стояла {T}", "мы уходили с {T} ещё в прошлом месяце"],
        "en": ["before that we ran {T}", "we moved off {T} last month"],
    },
    ("version", "hypothetical"): {
        "ru": ["может, откатиться на {T}?"],
        "en": ["should we roll back to {T}?"],
    },
    ("version", "mentioned_only"): {
        "ru": ["в анонсах обещали {T}", "это не про {T}"],
        "en": ["the announcement promised {T}", "this is not about {T}"],
    },
    ("host", "current"): {
        "ru": ["ломается узел {T}", "проблемный хост — {T}"],
        "en": ["the failing node is {T}", "the problem host is {T}"],
    },
    ("host", "historical"): {
        "ru": ["раньше это был {T}", "до переезда жили на {T}"],
        "en": ["it used to be {T}", "before the migration it was {T}"],
    },
    ("host", "mentioned_only"): {
        "ru": ["это не {T}", "документация указывает на {T}"],
        "en": ["it is not {T}", "the docs point at {T}"],
    },
    ("ip", "current"): {
        "ru": ["источник — {T}", "стучится с {T}"],
        "en": ["the source is {T}", "requests come from {T}"],
    },
    ("ip", "historical"): {
        "ru": ["раньше шло с {T}"],
        "en": ["traffic used to come from {T}"],
    },
    ("ip", "mentioned_only"): {
        "ru": ["но это не {T}"],
        "en": ["but it is not {T}"],
    },
    ("url", "current"): {
        "ru": ["падает запрос на {T}", "не открывается {T}"],
        "en": ["the request to {T} fails", "{T} does not open"],
    },
    ("url", "historical"): {
        "ru": ["раньше ломался {T}"],
        "en": ["{T} used to break"],
    },
    ("url", "mentioned_only"): {
        "ru": ["это не {T}"],
        "en": ["not {T}"],
    },
    ("uuid", "current"): {
        "ru": ["id инцидента {T}", "вот трейс: {T}"],
        "en": ["incident id {T}", "trace id: {T}"],
    },
    ("uuid", "historical"): {
        "ru": ["прошлый инцидент был под {T}"],
        "en": ["the previous incident was {T}"],
    },
    ("uuid", "mentioned_only"): {
        "ru": ["это не {T}"],
        "en": ["not {T}"],
    },
    ("path", "current"): {
        "ru": ["не доступен путь {T}", "нет прав на {T}"],
        "en": ["path {T} is unavailable", "no permissions for {T}"],
    },
    ("path", "historical"): {
        "ru": ["раньше лежало в {T}"],
        "en": ["it used to live under {T}"],
    },
    ("path", "mentioned_only"): {
        "ru": ["это не {T}"],
        "en": ["not {T}"],
    },
    ("port", "current"): {
        "ru": ["висит на порту {T}", "прослушивает {T}"],
        "en": ["it listens on port {T}", "bound to {T}"],
    },
    ("port", "mentioned_only"): {
        "ru": ["это не порт {T}"],
        "en": ["not port {T}"],
    },
    ("resource_id", "current"): {
        "ru": ["проблемный ресурс {T}", "падает {T}"],
        "en": ["the failing resource is {T}", "{T} keeps crashing"],
    },
    ("resource_id", "historical"): {
        "ru": ["раньше падал {T}"],
        "en": ["{T} used to crash"],
    },
    ("resource_id", "mentioned_only"): {
        "ru": ["это не {T}"],
        "en": ["not {T}"],
    },
    ("ticket_ref", "current"): {
        "ru": ["тикет {T} открыт", "продублирую заявку: {T}"],
        "en": ["ticket {T} is open", "duplicate request: {T}"],
    },
    ("ticket_ref", "historical"): {
        "ru": ["прошлый тикет {T} закрыли"],
        "en": ["the old ticket {T} was closed"],
    },
    ("ticket_ref", "mentioned_only"): {
        "ru": ["не этот тикет — {T} тут ни при чём"],
        "en": ["not this ticket — {T} is unrelated"],
    },
    ("datetime", "current"): {
        "ru": ["проблема началась {T}", "с {T} всё плохо"],
        "en": ["it started on {T}", "since {T} it is broken"],
    },
    ("datetime", "historical"): {
        "ru": ["раньше, {T}, уже было похожее"],
        "en": ["back on {T} we saw similar"],
    },
    ("datetime", "mentioned_only"): {
        "ru": ["это не {T}"],
        "en": ["not {T}"],
    },
}

# Noise lines: greetings, pleasantries, filler — NO technical literals, NO digits.
NOISE = {
    "ru": [
        "Добрый день!",
        "Помогите, пожалуйста.",
        "Заранее спасибо.",
        "Понял, спасибо.",
        "Проверим и вернёмся с ответом.",
        "Проблема очень мешает работе.",
        "Клиенты уже жалуются.",
        "С уважением, команда поддержки.",
        "Ждём вашего ответа.",
    ],
    "en": [
        "Hello!",
        "Please help.",
        "Thanks in advance.",
        "Got it, thanks.",
        "We will check and get back to you.",
        "This blocks our work.",
        "Customers are already complaining.",
        "Best regards, support team.",
        "Awaiting your reply.",
    ],
}

OPENERS = {
    "ru": ["Здравствуйте!", "Приветствую", "Добрый день"],
    "en": ["Hi there", "Hello", "Good day"],
}


def pick_frame(rng: random.Random, kind: LiteralKind, status: str, lang: str,
               has_component: bool) -> str:
    bank = KIND_FRAMES.get((kind, status), {}).get(lang) or GENERIC[status][lang]
    usable = [f for f in bank if has_component or "{C}" not in f]
    if not usable:
        usable = GENERIC[status][lang]
        usable = [f for f in usable if has_component or "{C}" not in f]
    return rng.choice(usable)


def fill(frame: str, token: str, component: str | None) -> str:
    out = frame.replace("{T}", token).replace("{C}", component or "")
    import re

    return re.sub(r" +", " ", out).strip()
