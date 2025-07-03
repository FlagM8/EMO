# NEAT Humanoid Walker – Evoluční modelování chodce

## Co program dělá

Simulace zobrazuje skupinu 50 jednoduchých „humanoidů“ složených z torza, hlavy, nohou a chodidel. Tyto postavy jsou poháněny motory v kloubech (kyčle, kolena, kotníky) a ovládány **umělými neuronovými sítěmi**, které se vyvíjejí pomocí algoritmu **NEAT (NeuroEvolution of Augmenting Topologies)**. 

![NEAT Humanoid Walker](https://raw.githubusercontent.com/FlagM8/EMO/refs/heads/main/emo.gif)

Cílem evoluce je:

- naučit humanoida stabilně stát a chodit,
- co nejvíce se posunout doprava (vpřed),
- udržovat rovnováhu a stabilní výšku hlavy,
- efektivně používat nohy k chůzi.

## Použité technologie

Projekt kombinuje několik knihoven a technologií:
- **Python** – hlavní programovací jazyk,
- **Pygame** – vizualizace prostředí, zobrazení chodců, neuronových sítí a statistiky,
- **Pymunk** – fyzikální engine pro realistickou simulaci těles, kloubů a pohybu,
- **NEAT-Python** – knihovna pro evoluci neuronových sítí s měnící se topologií.

## "Funkce"
- **Motory a omezení pohybu**: Každý kloub má motor a rotační limity, což zabraňuje „nepřirozeným“ pohybům (např. ohnutí kolena dozadu). - moc se stejně neosvědčilo :D
- **Sofistikovanější fitness funkce**: Hodnotí nejen vzdálenost, ale i stabilitu, výšku hlavy, rovnováhu, pohyb nohou a penalizuje nesprávné postoje (např. když se torso nebo hlava příliš přiblíží zemi). -musí se silně vyvážit pro optimální výsledek
- **„Smrtící zeď“**: Postava je postupně doháněna pohybující se stěnou, což zajišťuje tlak na postup vpřed a odstraňuje příliš pomalé jedince.
- **Vizualizace neuronové sítě**: Během běhu simulace se vykresluje struktura a spojení nejlepší neuronové sítě aktuální generace.

## Možná rozšíření
Do budoucna by bylo možné:
- přidat překážky do terénu (schody, nerovnosti),
- umožnit evoluci složitějších topologií (např. rekurentní sítě),
- zlepšit fyzikální model (přidat ruce, pružnější klouby),
- uložit a znovu přehrát chování vítězné postavy.
- editor postav
- vyvážení fitness funkce a přidání dalších "odměn"
- využití herního Enginu
- větší konfigurace
