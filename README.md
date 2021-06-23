# Projet Rooftop - Eté 2021
Ce répertoire est dédié à la Détection/Segmentation de toits goudronnés sur des images satellites en utilisant Mask R-CNN. Ce projet s'inscrit dans le cadre de ma participation à la compétition Copernicus Masters 2021. Mon but est d'extraire d'images satellites très hautes résolutions (l'échelle de la dizaine de centimètres) les toits d'entrepôts, d'entreprise ou de particuliers dont l'isolation est fait avec des matières très sombres qui emmagasinent énormément le rayonnement solaire ce qui provoque une surutilisation de la climatisation (ref société [CoolRoof](https://coolroof-france.com/en/)). La localisation des toits sur images satellites qui comportent des données géospatiales permet de générer des coordonnées GPS qui une fois envoyées à l'API de Google Maps permet de récupérer des données sur la structure (non, numéro de téléphone, adresse mail) via du scraping. 

## Outils utilisés 

- Dataset : [INRIA Aerial Image Labeling Dataset](https://project.inria.fr/aerialimagelabeling/) (resolution 0.3m) et [USGS](https://earthexplorer.usgs.gov/) NAIP via Google Earth Engine (resolution 0.6m)
- Etiquettage : [Labelme](https://github.com/wkentaro/labelme) Image Polygonal Annotation with Python
- Mask R-CNN : Modification du logiciel de [Matterport](https://github.com/matterport/Mask_RCNN) par [akTwelve](https://github.com/akTwelve/Mask_RCNN) pour être compatible avec Tensorflow 2.0



## Principe étudié 


Toit avec isolation seule            |  Toit avec isolation repeinte
:-------------------------:|:-------------------------:
<img src="https://github.com/vintel38/RoofTop/blob/master/doc/images/heat.png" width="300" /> | <img src="https://github.com/vintel38/RoofTop/blob/master/doc/images/cool.png"  width="300" />
Images issues de la [brochure](https://coolroof-france.com/wp-content/uploads/2021/05/plaquette_commerciale_en-1.pdf) de la société CoolRoof   

Le pipeline suivant, associé à des images satellites de très hautes résolutions, vise à établir une liste des bâtiments dont le toit pourrait être repeint avec des peintures hautement réflectives. En effet, cette technique permet de renvoyer une grande partie du rayonnement solaire vers l'espace ce qui a pour conséquence, d'après l'entreprise de peinture spéciale, de diminuer la température en surface du toit et en dessous du toit ce qui permet de limiter les effets délétères d'ilôts de chaleur et d'économiser de l'électricité sur le budget alloué à la climatisation. Ce projet est mené rapidement avec comme objectif une soumission à la compétition Copernicus Masters d'ici le 17 juillet 2021. 

## Pipeline réalisé


La concordance des coordonnées GPS avec des adresses physiques est assurée en utilisant l'API de Google Maps qui, moyennant finance, assigne les coordonnées physiques à celles GPS. 
