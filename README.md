# Predicting Medical Billing Codes from Clinical Notes (in MIMIC-III datasets) using BERT

Медицинское кодирование является неотъемлемой частью процесса выставления счетов за медицинские услуги в США. 
После осмотра пациента поставщику медицинских услуг необходимо заполнить платежную форму с соответствующими медицинскими кодами (болезнь или процедура) 
для получения платежей от страховых компаний или правительства.

В настоящее время этот процесс преобразования медицинских заметок со свободным текстом в стандартизированные медицинские коды в основном выполняется 
вручную либо самими клиницистами, либо профессиональными медицинскими программистами. Однако ручное кодирование требует много времени и подвержено ошибкам. 
В случае, когда медицинские счета за оказанные услуги будут отклонены, тем, кто участвует в этом процессе, будет трудно вспомнить, 
что произошло три-шесть месяцев назад, поэтому в дальнейшем могут возникнуть некоторые трудности с их оплатой, или же, скорее всего, не будут взысканы в судебном порядке в будущем. 
В результате точное и эффективное кодирование заметок в последние годы становится актуальной областью. Исследования по автоматическому кодированию появились в 
90-ые годы прошлого столетия.

Для решения этой преблемы мной создано приложение для дальнейшего предсказания медицинских кодов с помощью веб-фреймворка Streamlit, предназначенного для исследователей данных для 
простого развертывания моделей и визуализаций с использованием Python. В основу взяты специально предварительно натренированная модель BERT и токенайзер от
emilyalsentzer/Bio_ClinicalBERT для медицинских данных.