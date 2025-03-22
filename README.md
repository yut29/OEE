# OEE
Entwicklung einer OEE-Analyse-App für eine  Zerspanungsmaschine(Drehmaschine)mit Python und Streamlit

Diese Anwendung nutzt Computer Vision-Technologie und Deep Learning-Modelle, um Stillstände von Zerspanungsmaschinen zu erkennen und zu analysieren, wodurch die Produktionseffizienz verbessert werden kann.

## Funktionen

- Videoanalyse: Erkennung von Maschinenbetriebszuständen und verschiedenen Stillstandssituationen
- Zustandserkennung: Automatische Unterscheidung zwischen Arbeitszustand und verschiedenen Arten von Stillständen (wie Werkzeugjustierung, Reinigung usw.)
- Historische Daten: Aufzeichnung und Visualisierung detaillierter Verlaufsdaten der Maschinenzustände
- Effizienzanalyse: Berechnung und Darstellung von Verfügbarkeitskennzahlen und Stillstandsstatistiken

## Systemanforderungen

- **Python:** Python 3.10 empfohlen
- **Betriebssystem:** Die Anwendung wurde unter Windows entwickelt und getestet

## Installationsanleitung

### 1. Herunterladen
main.py und best.pt

### 2. Abhängigkeiten installieren
requirements.txt

### 3. Anwendung starten

```bash
streamlit run main.py
```

Die Anwendung wird in Standardbrowser geöffnet

## Bedienungsanleitung

1. **Videoquelle auswählen** - Sie können eine Videodatei(<200M) hochladen oder eine Kamera verwenden
2. **Videoanalyse** - Nach dem Hochladen eines Videos wird es automatisch analysiert und die Ergebnisse werden angezeigt
3. **Ergebnisse anzeigen** - Verwenden Sie die Registerkarten, um Analyseergebnisse, historische Daten und Verfügbarkeitsstatistiken anzuzeigen

## Parametereinstellungen

Die Seitenleiste der Anwendung bietet verschiedene Parametereinstellungen, die je nach spezifischer Maschine und Umgebung angepasst werden können:

- **Konfidenz-Schwellenwert:** Anpassung der Konfidenzanforderungen für die Objekterkennung
- **Werkzeug-Bewegungsschwelle:** Einstellung der Empfindlichkeit für die Werkzeugbewegungserkennung
- **Aufnahme-Helligkeitsänderungsschwelle:** Steuerung der Empfindlichkeit für die Erkennung von Helligkeitsänderungen
- **Innenraum-Frame-Differenzschwelle:** Einstellung der Parameter für die Erkennung von Bereichsänderungen
- **Frame-Verarbeitungsfrequenz:** Einstellung, wie oft Frames analysiert werden sollen
- **Erforderliche aufeinanderfolgende Frames für Statusänderung:** Steuerung der Stabilitätsanforderungen für Statusänderungen

