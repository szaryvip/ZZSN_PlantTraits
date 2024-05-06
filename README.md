# ZZSN_PlantTraits

Implementacja modelu sieci neuronowej do przewidywania cech numerycznych roślin na bazie dostarczonych zdjęć oraz danych tabelarycznych. Zadanie realizowane w formie konkursu na Kaggle w ramach projektu na przedmiocie ZZSN (Zaawansowane Zagadnienia Sieci Neuronowych) na Politechnice Warszawskiej: 
    

https://www.kaggle.com/competitions/planttraits2024/overview/abstract


Poniżej zostaną przedstawione dane, które należy przewidzieć. Nazwy zmiennych pozostaną w oryginale, aby nasze tłumaczenie nie pogorszyło możliwości zrozumienia wyszukiwanych danych. Każda ze zmiennych opisana jest swoim identyfikatorem, który realizowany jest przy pomocy specjalnego kodu umieszczonym przed myślnikiem


- X4 - Stem specific density (SSD) / wood density
- X11 - Leaf area per leaf dry mass
- X18 - Plant height
- X26 - Seed dry mass
- X50 - Leaf nitrogen
- X3112 - Leaf area


Celem jest przewidzenie wartości wyżej wymienionych zmiennych bazując na otrzymanych obrazach oraz danych warunkujących dostarczonych w postaci tabelarycznej.
W przypadku niektórych wartości zawartych w danych tabelarycznych zostały one sztucznie wygenerowane - proces ich generowania znajduje się w artykule: [Schiller, C., Schmidtlein, S., Boonman, C., Moreno-Martínez, A., \& Kattenborn, T. (2021). Deep learning and citizen science enable automated plant trait predictions from photographs. Scientific Reports, 11(1), 16395.](https://www.nature.com/articles/s41598-021-95616-0)

W celu poprawnego działania wszystkich skryptów, należy rozpakować dane uzyskane z platformy kaggle do folderu `data` w głównym folderze repozytorium.
