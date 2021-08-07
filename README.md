# Projet Rooftop - Eté 2021
Ce répertoire est dédié à la Détection/Segmentation/Localisation de toits goudronnés sur des images satellites en utilisant Mask R-CNN et le Google Maps API. Ce projet s'inscrit dans le cadre de ma participation à la compétition Copernicus Masters 2021. Mon but est d'extraire d'images satellites très hautes résolutions (résolution ~ 0.6m) les toits de bätiments dont l'isolation réalisée avec des matériaux sombres provoque une surutilisation de la climatisation (ref société [CoolRoof](https://coolroof-france.com/en/)). La localisation des toits sur images satellites avec données géospatiales produit des coordonnées GPS qui une fois envoyées à l'API de Google Maps conduit à des métadonnées sur les bâtiments. Les informations générées sont compilées dans une base de données de type PostGIS.

## Outils utilisés 

- Dataset : [INRIA Aerial Image Labeling Dataset](https://project.inria.fr/aerialimagelabeling/) (résolution 0.3m) et [USGS](https://earthexplorer.usgs.gov/) NAIP via Google Earth Engine (résolution 0.6m)
- Etiquettage : [Labelme](https://github.com/wkentaro/labelme) Image Polygonal Annotation with Python
- Mask R-CNN : Modification du logiciel de [Matterport](https://github.com/matterport/Mask_RCNN) par [akTwelve](https://github.com/akTwelve/Mask_RCNN) pour être compatible avec Tensorflow 2.0



## Principe étudié 


Toit avec isolation seule            |  Toit avec isolation repeinte
:-------------------------:|:-------------------------:
<img src="https://github.com/vintel38/RoofTop/blob/master/doc/images/heat.png" width="300" /> | <img src="https://github.com/vintel38/RoofTop/blob/master/doc/images/cool.png"  width="300" />
Images issues de la [brochure](https://coolroof-france.com/wp-content/uploads/2021/05/plaquette_commerciale_en-1.pdf) de la société CoolRoof   

Je me suis intéressé à la peinture de toit de bâtiments après avoir vu une vidéo de Loopsider sur le sujet sur le réseau Facebook. Le principe d'un point de vue technologique est assez simple pour être génial. 

Les bâtiments industriels de type entrepôts ont souvent l'isolation externe de leur toit réalisée avec des matériaux sombres composés d'hydrocarbures fondus au chalumeau pour assurer l'étanchéité. Le rayonnement solaire incident est alors majoritairement absorbé par le matériau. Ceci a pour effet d'augmenter drastiquement la température à la fois sur le toit mais aussi dans le bâtiment par conduction de l'énergie thermique dans les couches de matériau. Les effets délétères de ce phénomène sont donc une augmentation du phénomène d'ilôt de chaleur urbain ainsi qu'une utilisation plus prononcée de la climatisation qui participe également au phénomène d'ilôt de chaleur. La solution de CoolRoof est déjà connue depuis longtemps : les grecs repeignent depuis toujours leurs habitations en blanc ou encore la ville de New York a décidé de repeindre massivement les toits des bâtiments de sa ville. Repeindre le toit avec une peinture extrèmement réflective permet de renvoyer directement vers l'espace une part plus importante, voire majoritaire, du rayonnement solaire incident. Ainsi, on supprime le problème à la racine. Une part plus faible du rayonnement solaire incident demeure dans l'espace urbain ce qui limite le réchauffement de ce même espace. 

Ainsi, si l'élément clé de ce problème est l'aspect radiatif du toit qui est visible depuis l'espace d'où provient le rayonnement solaire, alors une solution utilisant de l'imagerie satellite thermique pour identifier les toits sujet à ce phénomène, fait sens. 


## Pipeline réalisé

La solution proposée dans ce projet utilise donc des images satellites très haute résolution (résolution ~ 0.6m) sur des longueurs visibles et infrarouge appelée CNIR (Common - Near InfraRed) qui peuvent traverser l'atmosphère sans trop d'altérations. Des images sur ces longueurs d'ondes intimement liées aux phénomènes thermiques permettent donc d'observer directement la thermodynamique d'objets au sol. L'algorithme de détection/segmentation Mask RCNN est donc théoriquement capable sur ces images de comprendre les implications thermiques sous-jacentes du problème et d'identifier/grouper les pixels qui appartiennent à des toits dont l'isolation externe doit être repeinte. Les images satellites utilisées sont de type GeoTiff avec un format sans perte d'informations sur l'imagerie qui embarque également des données géospatiales. Une fois que les toits sont identifiés individuellement, il devient possible de retrouver précisément leur positionnement GPS via leur position sur l'image et les données géospatiales associées à cette image. 

Finalement, ces coordonnées GPS sont envoyées à l'API de Google Maps qui moyennant finances est capable de vous donner des informations sur ce bâtiments comme l'adresse, le numéro de téléphone ou encore l'adresse mail du contact. Les données tirées de l'API de Google Maps sont alors compilées dans une base de données de type PostGIS qui représente alors la richesse et l'intérêt du projet RoofTop. En effet, les sociétés de peinture, les villes, les agglomérations ont tout intérêt à mettre la main sur une telle base de données qui signifie des économies substantielles en climatisation pour les bâtiments de leurs espaces et une amélioration des conditions de vie de la population avec un combat direct du phénomène d'ilôt de chaleur. 

<object data="https://github.com/vintel38/RoofTop-Project/blob/master/doc/Presentation2.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="https://github.com/vintel38/RoofTop-Project/blob/master/doc/Presentation2.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="https://github.com/vintel38/RoofTop-Project/blob/master/doc/Presentation2.pdf">Download PDF</a>.</p>
    </embed>
</object>