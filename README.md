# ocr_macos_polish
Wysoko wyspecjalizowany program do OCR polskich książek akademickich (nauki społeczne, humanistyczne, filozoficzne). Dla skanów robionych telefonem. Ogólnie to jest AI slope, czy vibe coding. Mam specyficzny workflow. Idę do biblioteki, biorę książkę, uruchamiam Iphona i korzystając z nowej funkcji (ios 26.x) skanuję dokumenty. Najczęściej wybieram żeby skanował jako dokumenty czarno-białe. Jest to opcja ekstemalnie szybka. Automatyczna detekcja stron działa całkiem dobrze, zatem robi zdjęcia i czeka na następną stronę. Ma tylko ograniczenie do 24 kartek. Taki skan, eksportuję do macbooka - łączę strony w pdf, naprawiam układ i daję do OCR.
Dotychczas używałem tesseracta i ocrmypdf. Było całkiem ok, ale chciałem sprawdzić czy natywny Apple Vison będzie lepiej wykrywał literki. W sumie ma przecież silnik do sieci neuronowych. Dlatego dla wydajności jest robiony w Swift. Zacząłem  robić program, został w wersji skryptu. Korzystałem z różnych narzędzi - Clauda, ChataGPT, wreszcie najwięcje na Gemini. To na czym chciałem się skupić, to etap przez LLM, czyli BERT, jako silnik rekomendacji słów, które sa niewykrywane. Działa całkiem nieźle. U mnie. Aha, licencja 
(## Licencja

Ten projekt jest licencjonowany na warunkach licencji **GNU General Public License v3.0 (GPL-3.0)**. ) 
# Apple Silicon PDF OCR Pipeline
A tu opis funkcji - generowany. 
Wielowątkowy, zoptymalizowany pod kątem sprzętowym (Apple Silicon) potok przetwarzania i korekty plików PDF. Narzędzie konwertuje surowe skany lub zdjęcia książek z aparatu do postaci standaryzowanych, przeszukiwalnych i zoptymalizowanych cyfrowo plików PDF, idealnych do czytania na tabletach (np. iPad Pro 13").

Projekt wykorzystuje natywne frameworki macOS (**Vision**, **CoreImage**, **CoreText** oraz **NaturalLanguage**) i jest w pełni akcelerowany sprzętowo przez systemowy silnik neuronowy (Apple Neural Engine).

---

## Główne Funkcjonalności

* **Standaryzacja do formatu A4 (Canvas Centering / Letterboxing):** Automatycznie i proporcjonalnie dopasowuje (Aspect Fit) i centruje każdą stronę na jednolitym wirtualnym płótnie formatu A4 (`595 x 842` punktów). Rozwiązuje to problem niespójnych wymiarów stron w PDF powstałych po docięciu zdjęć z telefonu.
* **Inteligentny podział rozkładówek (Spread Splitting):** Automatycznie lokalizuje fizyczny grzbiet/zgięcie książki i precyzyjnie dzieli horyzontalne rozkładówki na dwie osobne pionowe strony.
* **Precyzyjne pozycjonowanie tekstu (Word-by-word):** Nakłada niewidoczną warstwę tekstową słowo po słowie, dodając jawne spacje w strumieniu PDF. Zapobiega to pionowemu nakładaniu się zaznaczenia przy wygiętych liniach tekstu i chroni tekst przed uszkodzeniem przez Ghostscript.
* **Hybrydowa korekta diakrytyków Post-OCR:**
  * Korzysta z wątkowo bezpiecznej blokady systemowej `NSSpellChecker` z jawnym, dynamicznym wymuszeniem języka polskiego.
  * Posiada dedykowany moduł składania znaków, poprawnie traktujący polską literę „ł”/„Ł”.
  * Wykorzystuje **odległość edycyjną (Weighted Levenshtein)** jako twarde kryterium wyboru podpowiedzi.
* **Model BERT jako rozstrzygnięcie remisów (Tie-Breaker):** Przy remisach optycznych (np. wybór między *oddziałał* a *oddziałaj*) skrypt wstawia kandydatów do zdania próbnego i wykorzystuje natywny model językowy **BERT (`NLContextualEmbedding`)** do kontekstowego wyboru poprawnego słowa.
* **Ochrona nazw własnych:** Automatycznie wykrywa i chroni przed korektą nazwiska i nazwy własne przy użyciu `NLTagger`.
* **Ekstremalna optymalizacja RAM:** Cały zaawansowany proces językowy i graficzny mieści się w około **150 MB pamięci RAM**, dzięki czemu działa stabilnie nawet na maszynach z 8 GB pamięci zunifikowanej.

---

## Wymagania systemowe

* **System operacyjny:** macOS 14.0 (Sonoma) lub nowszy (macOS 15.0+ rekomendowany ze względu na wsparcie dla Super-Resolution i estymacji estetyki skanów).
* **Sprzęt:** Dowolny Mac z procesorem Apple Silicon (M1, M2, M3, M4) w celu użycia akceleracji Neural Engine.
* **Narzędzia deweloperskie:** Zainstalowany kompilator Swift (dostępny w pakiecie Xcode Command Line Tools).

---

## Instrukcja instalacji i konfiguracji

### Krok 1: Instalacja Command Line Tools
Upewnij się, że masz zainstalowany kompilator Swift na swoim komputerze Mac. Jeśli nie, zainstaluj go wpisując w terminalu:
```bash
xcode-select --install
### Krok 2: Aktywacja słownika języka polskiego w macOS
Aby post-OCR korekta językowa działała prawidłowo, upewnij się, że słownik polski jest aktywny w Twoim systemie macOS:

1. Otwórz **Ustawienia Systemowe -> Klawiatura**.
2. W sekcji **Wprowadzanie tekstu** kliknij przycisk **Edytuj...** obok pozycji *Metody wprowadzania* (Input Sources).
3. Znajdź opcję **Pisownia** (Spelling) i zmień ją z „Automatycznie wg języka” na sztywno na **„Polski”** (lub pobierz go z listy i zaznacz).

---

## Jak uruchomić program

### Opcja A: Uruchomienie ad-hoc (interpretacja w locie)
Możesz uruchomić skrypt bezpośrednio z poziomu terminala, podając ścieżkę do pliku wejściowego, wyjściowego oraz odpowiednie flagi:

```bash
swift ocr_pipeline.swift wejscie.pdf wyjscie.pdf --autocrop --fix-diacritics --nlp

Opcja B: Kompilacja dla maksymalnej wydajności (Zalecana)
Aby program przetwarzał strony z maksymalną prędkością, skompiluj go z flagą optymalizacji -O:

code
Bash
swiftc -O ocr_pipeline.swift -o ocr_pipeline
Po kompilacji uruchomisz go jako zwykły program binarny:

code
Bash
./ocr_pipeline wejscie.pdf wyjscie.pdf --autocrop --fix-diacritics --nlp
Opcja C: Dodanie programu do PATH systemowego (Globalne użycie)
Aby móc wywołać program z dowolnego miejsca w systemie pod nazwą ocr_pipeline, skopiuj skompilowany plik do lokalnego folderu binariów:

code
Bash
sudo cp ocr_pipeline /usr/local/bin/
Od teraz w dowolnym katalogu możesz wpisać:

code
Bash
ocr_pipeline wejscie.pdf wyjscie.pdf --autocrop --fix-diacritics --nlp
Dostępne flagi i parametry
--lang <l1,l2,...> — Języki OCR dla Vision (domyślnie pl-PL).

--dpi <liczba> — Rozdzielczość renderowania tła (domyślnie 300).

--autocrop — Włącza inteligentne kadrowanie i usuwanie czarnych ramek skanera.

--fix-diacritics — Włącza autorską korektę brakujących znaków diakrytycznych.

--nlp — Włącza zaawansowany model kontekstowy BERT (Ścieżka A) do rozstrzygania remisów pisowni.

--split-merged-words — Rozbija zlepione słowa przy użyciu słownika i programowania dynamicznego.

--nosplit — Wyłącza automatyczny podział rozkładówek (zachowuje oryginalny układ stron).

--color — Wymusza renderowanie tła w kolorze (sRGB) zamiast skali szarości.

--customwords <frazy/plik> — Słownik niestandardowych słów (np. nazwiska autorów), które mają być chronione przed korektą. Może to być lista po przecinku lub ścieżka do pliku .txt.

--sidecar <plik.txt> — Eksportuje czystą, zrekonstruowaną warstwę tekstową w formacie tekstowym.
