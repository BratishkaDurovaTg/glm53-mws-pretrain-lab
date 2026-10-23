"""T3: hand-written technical-support tickets (GLM-rendered, non-template origin).

Rules kept mechanical:
- every technical literal is a placeholder token (no raw codes/ids in text);
- each mention's declared status MUST match the wording (gold depends on it);
- typos/noise contain no technical literals.
"""

HANDWRITTEN = [
    {
        "id": "t3-001",
        "roles": ["user", "agent"],
        "texts": [
            "Добрый день. С субботы мобильный интернет отваливается, в приложении пишет ERR_1. "
            "Перезагрузки не помоглли, симку менял. Раньше уже было ERR_2, тогда само прошло.",
            "Здравствуйте! Проверим на нашей стороне. Уточните, пожалуйста, модель телефона.",
        ],
        "mentions": [
            ("ERR_1", "error_code", "current", None),
            ("ERR_2", "error_code", "historical", None),
        ],
        "selectors": [("error_code", "current"), ("error_code", "any")],
    },
    {
        "id": "t3-002",
        "roles": ["user"],
        "texts": [
            "Приставка на прошивке VER_1 работала нормально. После обновления до VER_2 начались "
            "зависания каждые минут десять. Откатил на VER_3 — пока полёт нормальный.",
        ],
        "mentions": [
            ("VER_1", "version", "historical", None),
            ("VER_2", "version", "historical", None),
            ("VER_3", "version", "current", None),
        ],
        "selectors": [("version", "current"), ("version", "any")],
    },
    {
        "id": "t3-003",
        "roles": ["user", "agent"],
        "texts": [
            "auth.log: connection refused from IP_1, повторяется каждые пять минут. "
            "Это не IP_2, он в белом списке.",
            "Спасибо, зафиксировали. Проверьте, пожалуйста, не менялся ли внешний адрес.",
        ],
        "mentions": [
            ("IP_1", "ip", "current", None),
            ("IP_2", "ip", "mentioned_only", None),
        ],
        "selectors": [("ip", "current"), ("ip", "any")],
    },
    {
        "id": "t3-004",
        "roles": ["user", "agent", "user"],
        "texts": [
            "Платёж не дошёл до лицевого счёта. Идентификатор операции UID_1, обращение в банке "
            "TKT_1. Деньги списались вчера.",
            "Проверим статус платежа, одну минуту.",
            "Спасибо, жду. Если важно — карта привязана к аккаунту UID_2.",
        ],
        "mentions": [
            ("UID_1", "uuid", "current", None),
            ("TKT_1", "ticket_ref", "current", None),
            ("UID_2", "uuid", "current", None),
        ],
        "selectors": [("uuid", "any"), ("ticket_ref", "current")],
    },
    {
        "id": "t3-005",
        "roles": ["user"],
        "texts": [
            "Здравствуйте! Подскажите, пожалуйста, как изменить тарифный план на другой? "
            "Заранее спасибо.",
        ],
        "mentions": [],
        "selectors": [("error_code", "any"), ("version", "any")],
    },
    {
        "id": "t3-006",
        "roles": ["user", "agent"],
        "texts": [
            "Не можем войти в личный кабинет, браузер показывает ERR_1. Трейс запроса: UID_1. "
            "Пробовали три браузера и инкогнито.",
            "Принято. Судя по трейсу, сессия падает на стороне аутентификации, передам "
            "инженерам. Ранее подобное лечилось очисткой куки.",
        ],
        "mentions": [
            ("ERR_1", "error_code", "current", None),
            ("UID_1", "uuid", "current", None),
        ],
        "selectors": [("error_code", "current"), ("uuid", "any")],
    },
    {
        "id": "t3-007",
        "roles": ["user", "agent"],
        "texts": [
            "Наш биллинг-коннектор по ночам падает: process RES_1 в статусе crash, в дампе "
            "упоминается модуль PTH_1. Днём всё ок.",
            "Посмотрели: ночная задача на узле HST_1 конфликтует с вашим процессом. "
            "Раньше это воспроизводилось на HST_2, но после переезда должно было уйти.",
        ],
        "mentions": [
            ("RES_1", "resource_id", "current", None),
            ("PTH_1", "path", "mentioned_only", None),
            ("HST_1", "host", "current", None),
            ("HST_2", "host", "historical", None),
        ],
        "selectors": [("resource_id", "current"), ("host", "current"), ("path", "any")],
    },
    {
        "id": "t3-008",
        "roles": ["user"],
        "texts": [
            "Webhook URL_1 timing out уже сутки, в логах вижу resource RES_1 в ошибке. "
            "Ретраи идут с адреса IP_1. Это началось после релиза VER_1 на нашей стороне, "
            "но откат до VER_2 не помог.",
        ],
        "mentions": [
            ("URL_1", "url", "current", None),
            ("RES_1", "resource_id", "current", None),
            ("IP_1", "ip", "current", None),
            ("VER_1", "version", "historical", None),
            ("VER_2", "version", "mentioned_only", None),
        ],
        "selectors": [("url", "current"), ("ip", "any"), ("version", "any")],
    },
    {
        "id": "t3-009",
        "roles": ["agent", "user"],
        "texts": [
            "Добрый день, дежурный инженер. По вашему обращению о недоступности панели: "
            "проблема воспроизводится на стенде HST_1, лечим. Наблюдайте за тикетом TKT_1.",
            "Спасибо! А смс-уведомления это тоже затронет?",
        ],
        "mentions": [
            ("HST_1", "host", "current", None),
            ("TKT_1", "ticket_ref", "current", None),
        ],
        "selectors": [("host", "current"), ("ticket_ref", "any")],
    },
    {
        "id": "t3-010",
        "roles": ["user", "agent", "user"],
        "texts": [
            "У нас в облаке упала виртуалка VM... то есть RES_1, простите, торопился. "
            "Не стартует, в консоли ERR_1. До этого стоял образ VER_1.",
            "Понял вас. Попробуйте стартовать с очисткой кэша, если ERR_2 появится — "
            "пришлите скриншот.",
            "Не, с кэшем то же самое. Новой ошибки нет, только та же самая снова.",
        ],
        "mentions": [
            ("RES_1", "resource_id", "current", None),
            ("ERR_1", "error_code", "current", None),
            ("VER_1", "version", "historical", None),
            ("ERR_2", "error_code", "hypothetical", None),
        ],
        "selectors": [("error_code", "current"), ("resource_id", "any")],
    },
    {
        "id": "t3-011",
        "roles": ["user"],
        "texts": [
            "Коллеги, привет. Хромает скорость до вашего объектного хранилища: скачивание с "
            "URL_1 идёт на 200 килобит, хотя раньше качалось с URL_2 на гигабит. "
            "Маршрут вроде не менялся.",
        ],
        "mentions": [
            ("URL_1", "url", "current", None),
            ("URL_2", "url", "historical", None),
        ],
        "selectors": [("url", "current"), ("url", "any")],
    },
    {
        "id": "t3-012",
        "roles": ["user", "agent"],
        "texts": [
            "Касса в магазине не пробивает чеки, на экране ERR_1. Оплата картой проходит, "
            "печать — нет. Фискальник FST... точнее RES_1 моргает красным.",
            "Проверим фискальный регистратор. Судя по индикации, проблема в прошивке: "
            "сейчас VER_1, а стабильная — VER_2. Откат возможен, согласны?",
        ],
        "mentions": [
            ("ERR_1", "error_code", "current", None),
            ("RES_1", "resource_id", "current", None),
            ("VER_1", "version", "current", None),
            ("VER_2", "version", "hypothetical", None),
        ],
        "selectors": [("version", "current"), ("error_code", "current")],
    },
    {
        "id": "t3-013",
        "roles": ["user"],
        "texts": [
            "Не открывается корпоративный портал, пишет ERR_1 со вчерашнего вечера. "
            "Из офиса и с мобильного интернета — одинаково. Коллеги из другого города "
            "говорят, что всё ок, значит дело не в нас.",
        ],
        "mentions": [
            ("ERR_1", "error_code", "current", None),
        ],
        "selectors": [("error_code", "current"), ("error_code", "any")],
    },
    {
        "id": "t3-014",
        "roles": ["user", "agent", "agent"],
        "texts": [
            "Добрый день! Хотел уточнить по заявке TKT_1 — уже неделю тишина. Там проблема "
            "с выгрузкой на PTH_1, ошибка ERR_1 в логах шедулера.",
            "Здравствуйте. Проверил: заявка в работе у вендора, дублирую номер: TKT_2.",
            "Добавил информацию в заявку, ожидайте ответа до конца дня.",
        ],
        "mentions": [
            ("TKT_1", "ticket_ref", "current", None),
            ("PTH_1", "path", "current", None),
            ("ERR_1", "error_code", "current", None),
            ("TKT_2", "ticket_ref", "current", None),
        ],
        "selectors": [("ticket_ref", "any"), ("path", "current")],
    },
    {
        "id": "t3-015",
        "roles": ["user"],
        "texts": [
            "Перестали приходить уведомления о списаниях. В приложении в разделе "
            "безопасности висит сессия с устройства HST_1, я её не узнаваю. "
            "Возможно, это старый планшет, но не уверен.",
        ],
        "mentions": [
            ("HST_1", "host", "hypothetical", None),
        ],
        "selectors": [("host", "any"), ("host", "current")],
    },
    {
        "id": "t3-016",
        "roles": ["user", "agent"],
        "texts": [
            "Мониторинг показывает рост ошибок на вашем API: за час ERR_1 повторился больше "
            "ста раз. Точка входа URL_1. При этом ERR_2 из той же панели — это наша "
            "внутренняя проверка, не ваши проблемы.",
            "Спасибо за разделение! По этой ошибке уже есть работа, статус можно смотреть "
            "в TKT_1.",
        ],
        "mentions": [
            ("ERR_1", "error_code", "current", None),
            ("URL_1", "url", "current", None),
            ("ERR_2", "error_code", "mentioned_only", None),
            ("TKT_1", "ticket_ref", "current", None),
        ],
        "selectors": [("error_code", "current"), ("url", "current")],
    },
    {
        "id": "t3-017",
        "roles": ["user"],
        "texts": [
            "После переезда офиса не работает телефония. АТС находится по адресу IP_1, "
            "трубки регистрируются, но звонки падают. Провайдер говорит, что порты "
            "открыты, кроме порта PRT_1, который они не открывают принципиально.",
        ],
        "mentions": [
            ("IP_1", "ip", "current", None),
            ("PRT_1", "port", "mentioned_only", None),
        ],
        "selectors": [("ip", "current"), ("port", "any")],
    },
    {
        "id": "t3-018",
        "roles": ["agent", "user", "agent"],
        "texts": [
            "Здравствуйте, вы оставляли обращение о проблемах с телевидением. "
            "Мы обновили софт приставки до VER_1, проверьте, пожалуйста, качество.",
            "Проверил — стало лучше, но на некоторых каналах артефакты остались.",
            "Хорошо, заведём дополнительный анализ, номер TKT_1. Диагностика будет "
            "завтра до DTE_1, вас устроит?",
        ],
        "mentions": [
            ("VER_1", "version", "current", None),
            ("TKT_1", "ticket_ref", "current", None),
            ("DTE_1", "datetime", "hypothetical", None),
        ],
        "selectors": [("version", "current"), ("datetime", "any")],
    },
    {
        "id": "t3-019",
        "roles": ["user"],
        "texts": [
            "Сайт партнёрского кабинета отдаёт ERR_1 при входе через ССО. Если зайти "
            "напрямую, минуя ССО, работает. В логах ССО вижу отказ для UID_1. "
            "В пятницу, DTE_1, всё ещё работало.",
        ],
        "mentions": [
            ("ERR_1", "error_code", "current", None),
            ("UID_1", "uuid", "current", None),
            ("DTE_1", "datetime", "historical", None),
        ],
        "selectors": [("error_code", "current"), ("uuid", "any"), ("datetime", "historical")],
    },
    {
        "id": "t3-020",
        "roles": ["user", "agent"],
        "texts": [
            "Утром пришло сообщение о блокировке аккаунта, код ERR_1. Я ничего не делал, "
            "подозреваю взлом. В письме была ссылка URL_1, но я по ней не переходил.",
            "Не переходите по ней. Заблокируем входы с HST_1, который фигурирует в "
            "журнале, и сменим пароль. Ожидайте инструкцию в TKT_1.",
        ],
        "mentions": [
            ("ERR_1", "error_code", "current", None),
            ("URL_1", "url", "mentioned_only", None),
            ("HST_1", "host", "current", None),
            ("TKT_1", "ticket_ref", "current", None),
        ],
        "selectors": [("host", "current"), ("url", "any"), ("ticket_ref", "current")],
    },
]
