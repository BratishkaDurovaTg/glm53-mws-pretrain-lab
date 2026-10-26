"""Natural-language frame bank (GLM-authored, Exp3). Casual/messy RU + RU/EN mix.

Same semantic contract as frames.py — wording MUST assert the mention's latent
status — but a completely different register: colloquial, lowercase, broken
punctuation, engineering slang. No digits/code outside {T}/{C} slots.
Provenance: teacher_model=glm53 (agent-authored static bank), see meta fields.
"""

BANK: dict[tuple[str, str], dict[str, list[str]]] = {
    ("_any", "current"): {
        "ru": [
            "щас вот ловим {T}",
            "опять {T}",
            "постоянно вылезает {T}",
            "сегодня уже n-й раз {T}",
            "клиенты видят {T}",
            "в проде сейчас {T}",
            "стабильно воспроизводится {T}",
        ],
        "en": [
            "right now we keep hitting {T}",
            "seeing {T} all day",
            "prod is throwing {T} again",
        ],
    },
    ("_any", "historical"): {
        "ru": [
            "вчера было {T}",
            "раньше ловили {T}",
            "до обновления было {T}",
            "на прошлой неделе {T}",
            "пару дней назад было {T}",
            "уже проходили с {T}",
        ],
        "en": [
            "yesterday we had {T}",
            "we used to get {T}",
            "before the update it was {T}",
        ],
    },
    ("_any", "resolved"): {
        "ru": [
            "{T} уже починили",
            "{T} само прошло",
            "после фикса {T} больше не было",
            "{T} — закрыли вчера",
        ],
        "en": [
            "{T} is fixed now",
            "{T} went away after the fix",
        ],
    },
    ("_any", "hypothetical"): {
        "ru": [
            "кажется это {T}",
            "вроде как {T}",
            "может быть {T}",
            "подозреваю {T}",
            "я бы поставил на {T}",
            "погуглил — похоже {T}",
        ],
        "en": [
            "maybe it is {T}",
            "I suspect {T}",
            "looks like {T} but not sure",
        ],
    },
    ("_any", "mentioned_only"): {
        "ru": [
            "это точно не {T}",
            "в доках пишут про {T}",
            "не путайте с {T}",
            "в соседнем чате обсуждали {T}, у нас такого нет",
            "реклама обещает {T}, но у нас не так",
            "раньше думал что {T}, но нет",
        ],
        "en": [
            "definitely not {T}",
            "the docs mention {T}",
            "do not confuse with {T}",
        ],
    },
    ("error_code", "current"): {
        "ru": [
            "после деплоя полетело с {T}",
            "шлюз отдаёт {T}",
            "при оплате падает с {T}",
            "апи отвечает {T}",
            "третий день {T}",
            "в логах {T} подряд",
            "{C} сыплет {T}",
        ],
        "en": [
            "{C} keeps returning {T}",
            "payments fail with {T}",
            "the endpoint throws {T}",
        ],
    },
    ("error_code", "historical"): {
        "ru": [
            "вчера {C} отдавал {T}",
            "до релиза ловили {T}",
            "в прошлый раз было {T}",
        ],
        "en": ["yesterday {C} returned {T}", "before the release we saw {T}"],
    },
    ("error_code", "resolved"): {
        "ru": ["{T} уже закрыли", "с {T} разобрались, больше не воспроизводится"],
        "en": ["{T} is resolved", "we fixed the {T} case"],
    },
    ("error_code", "hypothetical"): {
        "ru": ["похоже на {T}", "вроде эта ошибка {T}", "скорее всего {T}"],
        "en": ["looks like {T}", "probably {T}"],
    },
    ("error_code", "mentioned_only"): {
        "ru": ["это не {T}", "в changelog про {T} писали", "в факе указан {T}, не наш случай"],
        "en": ["not {T}", "the faq mentions {T}", "changelog says {T} — unrelated"],
    },
    ("version", "current"): {
        "ru": [
            "стоит {T}",
            "крутится {T}",
            "обновились до {T}",
            "щас на {T}",
            "в проде {T}",
        ],
        "en": ["we are on {T}", "running {T}", "upgraded to {T}"],
    },
    ("version", "historical"): {
        "ru": ["до этого стояла {T}", "была {T}", "ушли с {T} ещё вчера", "раньше жила {T}"],
        "en": ["we used to run {T}", "before that it was {T}"],
    },
    ("version", "hypothetical"): {
        "ru": ["может откатиться на {T}?", "предлагаю {T}", "думаю поставить {T}"],
        "en": ["should we roll back to {T}?", "proposing {T}"],
    },
    ("version", "mentioned_only"): {
        "ru": ["в анонсах {T}", "это не про {T}", "в релиз-нотах обещают {T}"],
        "en": ["the announcement promised {T}", "not about {T}"],
    },
    ("host", "current"): {
        "ru": ["лежит {T}", "проблемный узел {T}", "отваливается {T}", "{T} не отвечает"],
        "en": ["{T} is down", "the failing node is {T}"],
    },
    ("host", "historical"): {
        "ru": ["раньше это был {T}", "до переезда жили на {T}", "вчера ещё жил {T}"],
        "en": ["it used to be {T}", "before migration it was {T}"],
    },
    ("host", "resolved"): {
        "ru": ["{T} уже подняли", "{T} ожил после рестарта"],
        "en": ["{T} is back up"],
    },
    ("host", "hypothetical"): {
        "ru": ["возможно дело в {T}", "подозреваю {T}"],
        "en": ["maybe it is {T}"],
    },
    ("host", "mentioned_only"): {
        "ru": ["это не {T}", "доки ссылаются на {T}", "в примерах указан {T}"],
        "en": ["not {T}", "the docs point at {T}"],
    },
    ("ip", "current"): {
        "ru": ["стучится с {T}", "источник {T}", "блокирует {T}", "сессия с {T}"],
        "en": ["requests come from {T}", "the source is {T}"],
    },
    ("ip", "historical"): {
        "ru": ["раньше шло с {T}", "до смены было {T}"],
        "en": ["traffic used to come from {T}"],
    },
    ("ip", "mentioned_only"): {
        "ru": ["но это не {T}", "белый список тут не {T}"],
        "en": ["but it is not {T}"],
    },
    ("url", "current"): {
        "ru": ["падает {T}", "не открывается {T}", "хромается {T}", "тупит {T}"],
        "en": ["{T} is failing", "{T} does not open"],
    },
    ("url", "historical"): {
        "ru": ["раньше ломался {T}", "до редиректа жил {T}"],
        "en": ["{T} used to break"],
    },
    ("url", "mentioned_only"): {
        "ru": ["это не {T}", "реклама ведёт на {T}"],
        "en": ["not {T}"],
    },
    ("uuid", "current"): {
        "ru": ["трейс {T}", "инцидент {T}", "вот ид {T}", "сессия {T} висит"],
        "en": ["incident id {T}", "trace {T}"],
    },
    ("uuid", "historical"): {
        "ru": ["прошлый был {T}", "в прошлый раз {T}"],
        "en": ["the previous one was {T}"],
    },
    ("uuid", "mentioned_only"): {
        "ru": ["это не {T}", "из чата соседей {T}"],
        "en": ["not {T}"],
    },
    ("path", "current"): {
        "ru": ["нет прав на {T}", "не читается {T}", "падает на {T}", "конфиг {T}"],
        "en": ["{T} is unavailable", "it fails on {T}"],
    },
    ("path", "historical"): {
        "ru": ["раньше лежало в {T}", "до переноса был {T}"],
        "en": ["it used to live under {T}"],
    },
    ("path", "mentioned_only"): {
        "ru": ["это не {T}", "в примерах {T}"],
        "en": ["not {T}"],
    },
    ("port", "current"): {
        "ru": ["висит на {T}", "слушает {T}", "проброшен {T}"],
        "en": ["bound to {T}", "listening on {T}"],
    },
    ("port", "mentioned_only"): {
        "ru": ["это не {T}"], "en": ["not port {T}"],
    },
    ("resource_id", "current"): {
        "ru": ["падает {T}", "красный {T}", "в алертах {T}", "воркер {T} умирает"],
        "en": ["{T} keeps crashing", "alerts on {T}"],
    },
    ("resource_id", "historical"): {
        "ru": ["раньше падал {T}", "в прошлый раз {T}"],
        "en": ["{T} used to crash"],
    },
    ("resource_id", "mentioned_only"): {
        "ru": ["это не {T}", "соседний {T} не трогайте"],
        "en": ["not {T}"],
    },
    ("ticket_ref", "current"): {
        "ru": ["тикет {T} открыт", "заявка {T}", "дубль {T}"],
        "en": ["ticket {T} is open", "request {T}"],
    },
    ("ticket_ref", "historical"): {
        "ru": ["прошлый тикет {T} закрыли", "в марце был {T}"],
        "en": ["the old ticket {T} was closed"],
    },
    ("ticket_ref", "mentioned_only"): {
        "ru": ["не наш тикет {T}"], "en": ["unrelated ticket {T}"],
    },
    ("datetime", "current"): {
        "ru": ["с {T} всё плохо", "началось {T}", "уже {T} никакого ответа"],
        "en": ["since {T} it is broken", "started {T}"],
    },
    ("datetime", "historical"): {
        "ru": ["{T} уже было похожее", "ещё {T} лечили это"],
        "en": ["back on {T} we saw this"],
    },
    ("datetime", "mentioned_only"): {
        "ru": ["это не {T}"], "en": ["not {T}"],
    },
    ("money_amount", "current"): {
        "ru": ["списали {T}", "пришёл счёт на {T}", "задолженность {T}", "списание {T} снова"],
        "en": ["charged {T}", "the bill is {T}"],
    },
    ("money_amount", "historical"): {
        "ru": ["раньше было {T}", "в прошлом месяце {T}"],
        "en": ["it used to be {T}"],
    },
    ("money_amount", "hypothetical"): {
        "ru": ["обещают {T}", "хотят {T} за это"],
        "en": ["they promise {T}"],
    },
    ("money_amount", "mentioned_only"): {
        "ru": ["в рекламе {T}, по факту нет", "это не про {T}"],
        "en": ["the ad says {T}"],
    },
    ("phone_number", "current"): {
        "ru": ["мой номер {T}", "звоните на {T}", "симка {T}"],
        "en": ["my number is {T}"],
    },
    ("phone_number", "historical"): {
        "ru": ["раньше был {T}", "старый номер {T} уже не мой"],
        "en": ["the old number was {T}"],
    },
    ("phone_number", "mentioned_only"): {
        "ru": ["не звоните на {T}"], "en": ["not {T}"],
    },
    ("account_number", "current"): {
        "ru": ["л/с {T}", "лицевой {T}", "счёт {T}"],
        "en": ["account {T}"],
    },
    ("account_number", "historical"): {
        "ru": ["был л/с {T}", "прежний счёт {T}"],
        "en": ["the previous account was {T}"],
    },
    ("account_number", "mentioned_only"): {
        "ru": ["это не {T}"], "en": ["not {T}"],
    },
    ("order_id", "current"): {
        "ru": ["заказ {T}", "мой ord {T}", "доставка по {T}"],
        "en": ["order {T}"],
    },
    ("order_id", "historical"): {
        "ru": ["прошлый заказ был {T}"],
        "en": ["the previous order was {T}"],
    },
    ("order_id", "mentioned_only"): {
        "ru": ["не этот заказ — {T}"], "en": ["not order {T}"],
    },
    ("application_id", "current"): {
        "ru": ["заявка {T}", "обращение {T}"],
        "en": ["application {T}"],
    },
    ("application_id", "historical"): {
        "ru": ["в прошлый раз заявка была {T}"],
        "en": ["the earlier application was {T}"],
    },
    ("application_id", "mentioned_only"): {
        "ru": ["это не {T}"], "en": ["not {T}"],
    },
    ("transaction_id", "current"): {
        "ru": ["транзакция {T}", "чек {T}", "платёжка {T}"],
        "en": ["transaction {T}"],
    },
    ("transaction_id", "historical"): {
        "ru": ["прошлый платёж был {T}"],
        "en": ["the previous transaction was {T}"],
    },
    ("transaction_id", "mentioned_only"): {
        "ru": ["не та операция — {T}"], "en": ["not {T}"],
    },
    ("tariff_name", "current"): {
        "ru": ["тариф {T}", "сижу на {T}", "в приложении {T}"],
        "en": ["I am on {T}"],
    },
    ("tariff_name", "historical"): {
        "ru": ["раньше был {T}", "ушёл с {T}"],
        "en": ["I used to be on {T}"],
    },
    ("tariff_name", "hypothetical"): {
        "ru": ["хочу {T}", "предлагают {T}", "думаю про {T}"],
        "en": ["considering {T}", "they offered {T}"],
    },
    ("tariff_name", "mentioned_only"): {
        "ru": ["в рекламе {T}", "это не {T}"],
        "en": ["the ad is about {T}"],
    },
    ("service_name", "current"): {
        "ru": ["подключена {T}", "пришла {T}", "опять {T}"],
        "en": ["{T} is active", "subscribed to {T}"],
    },
    ("service_name", "historical"): {
        "ru": ["раньше была {T}", "отключал {T} ещё зимой"],
        "en": ["I used to have {T}"],
    },
    ("service_name", "hypothetical"): {
        "ru": ["хотите подключить {T}?", "предлагаю {T}"],
        "en": ["how about {T}?"],
    },
    ("service_name", "mentioned_only"): {
        "ru": ["реклама про {T}", "это не {T}"],
        "en": ["the ads are about {T}"],
    },
    ("device_name", "current"): {
        "ru": ["телефон {T}", "аппарат {T}", "на {T} воспроизводится"],
        "en": ["my device is {T}", "reproduces on {T}"],
    },
    ("device_name", "historical"): {
        "ru": ["раньше был {T}"],
        "en": ["the old one was {T}"],
    },
    ("device_name", "mentioned_only"): {
        "ru": ["это не {T}"], "en": ["not {T}"],
    },
}

NOISE_NATURAL = {
    "ru": [
        "добрый день",
        " привет",
        "уже третий раз пишу",
        "спасибо заранее",
        "срочно!!",
        "извиняюсь если не туда",
        "прошу прощения за панику",
        "жду уже неделю",
        "что делать?",
        "скажите пожалуйста",
        "надеюсь на вас",
        "писал вчера в другой чат, тишина",
        "перечитайте мой вопрос пожалуйста",
        "заранее благодарен",
    ],
    "en": [
        "hello",
        "please help",
        "third time asking",
        "thanks in advance",
        "urgent",
        "sorry for the spam",
        "any update?",
        "really frustrated here",
        "waiting for a week already",
        "kind regards",
    ],
}
