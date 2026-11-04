# Wytyczne anotacji, etap 2: niepewność pracy, branża i skutki AI dla pracy (Wykop)

Pogłębiona anotacja po pierwszej analizie ([etap 1](annotation-guidelines.md)). Oceniasz **wszystkie teksty, które w etapie 1 mają `work` = yes**, niezależnie od tego, czy wspominają o AI. Przed 2023 rokiem tekstów o AI i pracy prawie nie ma, a chcemy sprawdzić, czy przed premierą ChatGPT ludzie też bali się o swoją pracę i z jakich powodów; dlatego niepewność pracy oceniamy w każdym tekście o pracy. Zasady ogólne (oceniasz tylko to, co autor napisał, nie domyślasz się niewyrażonych opinii, hashtagi nie są dowodem, ironię oceniasz według zamierzonego sensu, wątpliwości rozstrzygasz najbezpieczniejszą etykietą) są takie same jak w etapie 1. Widzisz tylko treść tekstu.

Etykiety, w tej kolejności:

1. dla każdego tekstu: `job_insecurity`, `insecurity_cause`, `sector`;
2. o AI: `stance`, `attitude`, `threat_type`, `competence_claim`. Jeśli tekst w ogóle nie dotyczy AI (te same kryteria co `ai` w etapie 1), wszystkie cztery zostają puste.

## `job_insecurity`: niepewność pracy

Dla każdego tekstu, także bez AI. Czy autor wyraża obawę o bezpieczeństwo pracy, dochód albo możliwość znalezienia pracy, swoją lub innych, niezależnie od jej źródła. **yes**: strach przed zwolnieniem lub zastąpieniem, przed brakiem pracy albo zatrudnienia, przed zbyt niskimi lub spadającymi zarobkami, przed zanikiem zawodu lub branży; także rezygnacja, ironia albo rada "przebranżów się", jeśli wyraźnie niosą taką obawę. **no**: neutralne informacje, pytania, porady, narzekanie na szefa, zespół czy ilość pracy bez obawy o etat lub dochód, pewność siebie, zadowolenie. Wątpliwości: `no`.

## `insecurity_cause`: źródło niepewności

Tylko gdy `job_insecurity` = yes, w pozostałych przypadkach pole zostaje puste. Zapisuje źródło, które wskazuje autor; przy kilku przyczynach wybierz tę, którą autor najmocniej podkreśla.

| Wartość                  | Kiedy                                                                                   |
| ------------------------ | --------------------------------------------------------------------------------------- |
| `ai`                     | AI lub automatyzacja zastępuje pracę albo zmniejsza popyt na nią                        |
| `economy`                | sytuacja gospodarcza, kryzys, inflacja, recesja, zwolnienia w branży lub na całym rynku ("kryzys w IT", "zwolnienia wszędzie") |
| `offshoring_immigration` | przenoszenie pracy za granicę albo konkurencja pracowników z zagranicy                  |
| `employer`               | praktyki pracodawcy: restrukturyzacja, umowy, likwidacja firmy, niewypłacone pensje, mobbing zagrażający posadzie |
| `individual`             | sytuacja autora lub danej osoby: wiek, brak umiejętności lub doświadczenia, zdrowie, wykształcenie, miejsce zamieszkania |
| `other`                  | inne wskazane źródło                                                                    |
| `unstated`               | obawa bez podanej przyczyny                                                             |

`ai` jako przyczyna nie wymaga, by autor oceniał AI; wystarczy, że AI lub automatyzację wskazuje jako źródło obawy.

## `sector`: branża lub zawód, o który chodzi

Dla każdego tekstu, także bez AI i bez niepewności. Główna dziedzina zawodowa, o której jest tekst, tak jak pisze o niej autor. Jeśli autor wymienia kilka branż, wybierz tę, której dotyczy główna myśl tekstu. Sam hashtag nie wskazuje branży.

| Wartość       | Kiedy                                                                     |
| ------------- | ------------------------------------------------------------------------- |
| `it`          | programowanie, IT, testowanie, administracja systemami                    |
| `creative`    | grafika, ilustracja, teksty, tłumaczenia, muzyka, film                    |
| `office`      | praca biurowa, administracja, księgowość, urzędy                          |
| `service`     | obsługa klienta, sprzedaż, call center                                    |
| `education`   | nauczanie, szkolnictwo, nauka                                             |
| `other`       | inna wskazana branża lub zawód (np. budowlanka, medycyna, transport, produkcja) |
| `unspecified` | autor nie wskazuje branży ani zawodu, mówi ogólnie o rynku pracy          |

## `stance`: skutki AI dla pracy

Mówi, co autor twierdzi, że AI oznacza dla pracy. Etykietę inną niż `none` dajesz tylko wtedy, gdy autor wyraźnie łączy AI z pracą.

| Wartość       | Kiedy                                                                                                                                             |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `threat`      | AI odbierze pracę, ograniczy zatrudnianie (też juniorów), obniży zarobki, zdezaktualizuje zawód; strach, rezygnacja, rada zmiany zawodu. Także ironicznie. |
| `reassurance` | prace są bezpieczne, strach przesadzony, AI za słabe, by zastąpić ludzi                                                                           |
| `opportunity` | AI pomaga pracownikom: produktywność, nowe zawody, wyższe zarobki, przewaga w karierze                                                            |
| `mixed`       | jednocześnie zagrożenie i szansa lub uspokojenie                                                                                                  |
| `none`        | brak opinii o skutkach AI dla pracy                                                                                                               |

Co jest `none`:

- wypalenie, zarobki, szukanie pracy, gdy tekst wspomina AI, ale nie łączy go z pracą,
- newsy o produktach i cenach AI,
- tekst wygenerowany przez AI i wklejony bez komentarza (to nie jest pogląd autora),
- samo wklejenie nagłówka newsa (liczy się tylko, gdy autor się zgadza lub reaguje),
- samo "używam ChatGPT w pracy" bez oceny; "AI pomaga mi w pracy" to już `opportunity`.

Rozstrzyganie wątpliwości przy `stance`:

1. Brak jasnego związku AI z pracą: `none`.
2. Dwie sprzeczne perspektywy: `mixed`.
3. Pytanie retoryczne bez odpowiedzi autora: `none`; jeśli ton wyraźnie wyraża obawę: `threat`.
4. Opis cudzych obaw bez oceny autora: `none`; jeśli autor je potwierdza: odpowiednia etykieta.
5. Zawsze sprawdź, czyj to pogląd: autora czy cytowanej osoby.

Na Wykopie ironia jest częsta ("kryzysit", "rip programiści"). Ironiczny strach to nadal `threat`.

## `attitude`: nastawienie do AI na rynku pracy

`stance` mówi, co AI oznacza dla pracy, a `attitude` ocenia nastawienie autora do obecności AI na rynku pracy: czy wyraża się o niej dobrze, czy źle.

| Wartość    | Kiedy                                                                                          |
| ---------- | ---------------------------------------------------------------------------------------------- |
| `positive` | AI na rynku pracy oceniane dobrze: pomocne, przydatne, budzi entuzjazm lub nadzieję            |
| `negative` | AI na rynku pracy oceniane źle: groźne, szkodliwe, bezwartościowe, budzi lęk, złość lub drwinę |
| `mixed`    | w tekście są jednocześnie pozytywne i negatywne oceny                                          |
| `neutral`  | autor opisuje lub komentuje AI na rynku pracy bez oceny                                        |
| puste      | tekst nie dotyczy AI (patrz wyżej)                                                             |

Ocena nie zawsze pokrywa się ze `stance`: ktoś może obawiać się utraty pracy (`threat`), a mimo to dobrze oceniać obecność AI na rynku pracy (`positive`). Ironię oceniasz według zamierzonego sensu.

## `threat_type`: rodzaj obawy

Tylko gdy `stance` to `threat` albo `mixed`. W pozostałych przypadkach pole zostaje puste.

| Wartość    | Kiedy                                                                   |
| ---------- | ----------------------------------------------------------------------- |
| `job_loss` | AI zabierze komuś pracę, zwolnienia, likwidacja stanowisk               |
| `hiring`   | trudniej znaleźć pracę lub wejść na rynek, np. mniej ofert dla juniorów |
| `pay`      | niższe zarobki lub stawki                                               |
| `skills`   | umiejętności lub zawód się zdezaktualizują                              |
| `other`    | inna obawa o pracę, której nie obejmują powyższe                        |

Jeśli tekst zawiera kilka obaw, wybierz tę, którą autor akcentuje najmocniej, a przy równorzędnych tę, która pojawia się pierwsza.

## `competence_claim`: czy AI zmienia wymagane kompetencje

Zapisuje, **co autor twierdzi** o wymaganiach wobec pracowników, a nie to, jak jest w rzeczywistości.

| Wartość     | Kiedy                                                                                           |
| ----------- | ----------------------------------------------------------------------------------------------- |
| `more`      | AI wymaga nowych lub dodatkowych umiejętności ("trzeba umieć obsługiwać AI", "wymagania rosną") |
| `fewer`     | AI zmniejsza wymagania lub zastępuje umiejętności ("nie trzeba już umieć kodować")              |
| `unchanged` | autor wprost twierdzi, że wymagania się nie zmieniają                                           |
| `none`      | autor nie mówi o kompetencjach lub umiejętnościach                                              |

Jeśli autor twierdzi jedno i drugie, wybierz to, co akcentuje mocniej. Samo "używam AI w pracy" bez wniosku o wymaganiach to `none`.

## Przykłady

Kolejność: `job_insecurity`, `insecurity_cause`, `sector`, `stance`, `attitude`, `threat_type`, `competence_claim` ("–" to puste pole).

| Tekst | Etykiety |
| ----- | -------- |
| "Wysłałem 200 CV i cisza, z tym kryzysem w IT chyba nigdy nie znajdę roboty" | yes, `economy`, `it`, –, –, –, – |
| "Mam 52 lata i boję się, że po zwolnieniu nikt mnie już nie przyjmie na magazyn" | yes, `individual`, `other`, –, –, –, – |
| "Kolejny dzień w korpo, PM znowu przesunął deadline" | no, –, `office`, –, –, –, – |
| "Za rok juniorów nikt nie zatrudni, Copilot pisze kod szybciej" | yes, `ai`, `it`, `threat`, `negative`, `hiring`, `fewer` |
| "Dzięki ChatGPT robię raporty w księgowości w godzinę, polecam" | no, –, `office`, `opportunity`, `positive`, –, `none` |
| "AI odbierze pracę tłumaczom, ale kto nauczy się z nim pracować, zarobi więcej" | yes, `ai`, `creative`, `mixed`, `mixed`, `job_loss`, `more` |
| "To tylko lepsze autouzupełnianie, nikogo w poważnej pracy nie zastąpi" | no, –, `unspecified`, `reassurance`, `negative`, –, `none` |
| "Firma kupiła nam licencje na Copilota, od poniedziałku szkolenie" | no, –, `unspecified`, `none`, `neutral`, –, `none` |

## Format

Dla każdego tekstu, w tej kolejności: `job_insecurity` (yes/no), `insecurity_cause` (ai/economy/offshoring_immigration/employer/individual/other/unstated albo puste), `sector` (it/creative/office/service/education/other/unspecified), `stance` (threat/reassurance/opportunity/mixed/none), `attitude` (positive/negative/mixed/neutral), `threat_type` (job_loss/hiring/pay/skills/other albo puste) i `competence_claim` (more/fewer/unchanged/none). Dla tekstu bez AI `stance`, `attitude`, `threat_type` i `competence_claim` są puste.
