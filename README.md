## Customer Segmentation API

### Opis
Customer Segmentation API to aplikacja webowa napisana przy użyciu FastAPI, która umożliwia klasyfikację klientów na podstawie różnych cech demograficznych i behawioralnych. Projekt wykorzystuje model Random Forest do dokonywania predykcji, a także oblicza SHAP (SHapley Additive exPlanations) wartości dla każdej zmiennej wejściowej.


### O Zestawie Danych

Segmentacja klientów to praktyka dzielenia bazy klientów na grupy osób podobnych pod względem określonych cech istotnych z punktu widzenia marketingu, takich jak wiek, płeć, zainteresowania i nawyki zakupowe.

Firmy stosujące segmentację klientów działają w oparciu o fakt, że każdy klient jest inny, i że ich wysiłki marketingowe będą skuteczniejsze, jeśli skierują je do konkretnych, mniejszych grup osób, z wiadomościami, które ci konsumenci uznają za istotne i skłonią ich do zakupu czegoś. Firmy również mają nadzieję na uzyskanie głębszego zrozumienia preferencji i potrzeb swoich klientów, aby odkryć, co każdy segment uważa za najbardziej wartościowe, aby dokładniej dostosować materiały marketingowe do tego segmentu.


Firma motoryzacyjna planuje wejście na nowe rynki ze swoimi istniejącymi produktami (P1, P2, P3, P4 i P5). Po intensywnych badaniach rynku wywnioskowali, że zachowanie nowego rynku jest podobne do ich istniejącego rynku.

Na istniejącym rynku zespół sprzedażowy sklasyfikował wszystkich klientów na 4 segmenty (A, B, C, D). Następnie przeprowadzili zróżnicowane działania i komunikację dla różnych segmentów klientów. Ta strategia sprawdziła się wyjątkowo dobrze. Planują zastosować tę samą strategię na nowych rynkach i zidentyfikowali 2627 potencjalnych nowych klientów.

Masz za zadanie pomóc menedżerowi w przewidzeniu prawidłowej grupy nowych klientów.

### Instalacja
Aby zainstalować i uruchomić aplikację, wykonaj następujące kroki:

1. Sklonuj repozytorium na swój lokalny komputer:

git clone https://github.com/your-username/customer-segmentation-api.git


2. Przejdź do katalogu projektu:

cd customer-segmentation-api

3. Zainstaluj wymagane biblioteki Pythona za pomocą pip:

pip install -r requirements.txt

4. Uruchom aplikację:

uvicorn main:app --reload

Po wykonaniu tych kroków, aplikacja będzie dostępna pod adresem http://localhost:8000.

## Użycie

### Interakcja z API

Aplikacja udostępnia endpoint /convert_to_numpy, który obsługuje zapytania POST. Możesz przesłać dane klienta do tego endpointu, a otrzymasz predykcję dotyczącą segmentacji klienta oraz SHAP wartości dla każdej zmiennej wejściowej.

Przykładowe zapytanie CURL:

curl -X 'POST' \
  'http://localhost:8000/convert_to_numpy' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "Gender": 1,
  "Ever_Married": 1,
  "Age": 35,
  "Graduated": 1,
  "Work_Experience": 5,
  "Spending_Score": 2,
  "Family_Size": 4,
  "Profession_Artist": 0,
  "Profession_Doctor": 0,
  "Profession_Engineer": 1,
  "Profession_Entertainment": 0,
  "Profession_Executive": 0,
  "Profession_Healthcare": 0,
  "Profession_Homemaker": 0,
  "Profession_Lawyer": 0,
  "Profession_Marketing": 0,
  "Var_1_Cat_1": 0,
  "Var_1_Cat_2": 1,
  "Var_1_Cat_3": 0,
  "Var_1_Cat_4": 0,
  "Var_1_Cat_5": 0,
  "Var_1_Cat_6": 0,
  "Var_1_Cat_7": 0
}'

### Wyniki zapytania

Po wysłaniu zapytania otrzymasz odpowiedź w formacie JSON zawierającą przewidywaną segmentację klienta oraz SHAP wartości dla każdej zmiennej wejściowej.

Przykładowa odpowiedź:

{
  "segmentation": "b",
  "shap_values": {
    "Gender": -0.1,
    "Ever_Married": 0.2,
    "Age": -0.05,
    "Graduated": 0.1,
    "Work_Experience": -0.15,
    "Spending_Score": 0.25,
    "Family_Size": -0.08,
    "Profession_Artist": 0.02,
    "Profession_Doctor": -0.03,
    "Profession_Engineer": 0.05,
    "Profession_Entertainment": 0.1,
    "Profession_Executive": 0.12,
    "Profession_Healthcare": -0.07,
    "Profession_Homemaker": 0.08,
    "Profession_Lawyer": -0.1,
    "Profession_Marketing": 0.06,
    "Var_1_Cat_1": -0.04,
    "Var_1_Cat_2": 0.03,
    "Var_1_Cat_3": -0.02,
    "Var_1_Cat_4": 0.01,
    "Var_1_Cat_5": 0.02,
    "Var_1_Cat_6": -0.03,
    "Var_1_Cat_7": 0.04
  }
}

### Autor

Michalina Czechowska
Bartosz Rybiński
Łukasz Kałamarski
Michał Walendzewicz