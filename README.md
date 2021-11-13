# Projet Rooftop - Eté 2021
Ce répertoire est dédié à la Détection/Segmentation/Localisation de toits goudronnés sur des images satellites en utilisant Mask R-CNN et le Google Maps API. Ce projet s'inscrit dans le cadre de ma participation à la compétition Copernicus Masters 2021. Mon but est d'extraire d'images satellites très hautes résolutions (résolution ~ 0.6m) les toits de bätiments dont l'isolation réalisée avec des matériaux sombres provoque une surutilisation de la climatisation (ref société [CoolRoof](https://coolroof-france.com/en/)). La localisation des toits sur images satellites avec données géospatiales produit des coordonnées GPS qui une fois envoyées à l'API de Google Maps conduit à des métadonnées sur les bâtiments. Les informations générées sont compilées dans une base de données de type PostGIS.

## Outils utilisés 

- Dataset : [INRIA Aerial Image Labeling Dataset](https://project.inria.fr/aerialimagelabeling/) (résolution 0.3m) et [USGS](https://earthexplorer.usgs.gov/) NAIP via Google Earth Engine (résolution 0.6m)
- Etiquettage : [Labelme](https://github.com/wkentaro/labelme) Image Polygonal Annotation with Python
- Mask R-CNN : Modification du logiciel de [Matterport](https://github.com/matterport/Mask_RCNN) par [akTwelve](https://github.com/akTwelve/Mask_RCNN) pour être compatible avec Tensorflow 2.0

## Article 

Vous pouvez retrouver un article que j'ai écrit en anglais sur le thread de Towards Data Science un peu plus détaillé à propos de ce projet [ici](https://towardsdatascience.com/my-rooftop-project-a-satellite-imagery-computer-vision-example-e45a296129a0?source=social.linkedin&_nonce=bHKuLyjO). 