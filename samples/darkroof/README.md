# RoofTop Project - Compte-rendu de développement 

Ce petit fichier vise à vous expliquer le cheminement du développement du pipeline RoofTop après la soumission du dossier de candidature aux   Copernicus Masters le 17 juillet 2021. Comme cela peut être remarqué dans le pdf de présentation du projet, le pipeline se compose en deux parties majeures : 

- La détection / instance segmentation des toits sombres par l'algorithme Mask RCNN qui exploite les propriétés graphiques des toits (intrinsèquement liées à leurs propriétés thermiques) pour localiser individuellement ceux qui pourraient être repeints. 
- La génération de métadonnées pour ces toits sombres par l'API de Google Maps en utilisant leurs coordonnées GPS. 

## <ins> Phase 1 : Juillet 2021 - Essais initiaux </ins>

La première phase de développement a été menée en utilisant mes capacités de programmation à l'instant t ainsi que celles trouvées sur internet. Dans l'ordre des choses et en estimant la difficulté de mise en oeuvre, j'ai préféré me concentrer sur l'algorithme d'IA pour mettre en marche le projet RoofTop. En effet, les API de Google sont souvent très simple à utiliser et très ergonomique. La page GitHub de l'implémentation de Mask RCNN par [Matterport](https://github.com/matterport/Mask_RCNN) recense des informations sur les cas d'utilisation du logiciel notamment avec de l'imagerie satellite. La page de [Mstfakts](https://github.com/Mstfakts/Building-Detection-MaskRCNN) propose une utilisation de Mask RCNN pour de la détection de bâtiments ainsi qu'un workflow. Le pipeline de la phase 1 s'inspire donc largement de ces techniques. 

- Fixer la résolution à 0.6 m par pixel entre les datasets utilisés. Script : [preprocess.py](https://github.com/vintel38/RoofTop-Project/blob/master/doc/preprocess.py) effectué basiquement avec la fonction `cv2.resize`
- Découper les images 5000x5000 px d'origine en images 1024x1024 px, taille optimale pour l'entraînement de Mask RCNN. Script : [preprocess.py](https://github.com/vintel38/RoofTop-Project/blob/master/doc/preprocess.py) effectué basiquement avec l'extraction de tableaux de numpy
- Ecriture de l'image résultante en image PNG d'après le post de Mstfakts. Script : [preprocess.py](https://github.com/vintel38/RoofTop-Project/blob/master/doc/preprocess.py) effectué basiquement avec la fonction `cv2.imwrite`

L'entraînement est réalisé sur Google Colab en profitant de l'accélération machine des GPU Nvidia Tesla K80 sans quoi le temps de calcul sur CPU devient prohibitif. Les fichiers sources sont uploadés sur Github et seul un notebook est uploadé sur Google Drive. Le notebook permet au lancement de télécharger l'intégralité des fichiers utiles sur le Github puis de les utiliser. 

Ce pipeline n'a pas permis de faire fonctionner l'algorithme Mask RCNN, du moins d'avoir des détections en sortie de la procédure de test. De plus, de nombreuses incohérences ont été mis en lumière pour un tel pipeline vis-à-vis de ses objectifs globaux. 

- Les données géospatiales contenues dans le GeoTIFF sont perdues pendant la conversion du `.tif` vers le `.png`. Or elles sont primordiales dans la suite du projet pour localiser les toits sombres détectés par Mask RCNN dans des coordonnées GPS globales par le Google Maps API. Des solutions existent en convertissant le `.tif` vers le `.png` via des [fonctions](https://www.programmersought.com/article/48811569657/) de la bibliothèque GDAL en produisant un `.png` plus un fichier `.png.aux` annexe contenant les informations géospatiales associées. Un exemple est implémenté dans le script Python [`cvtTIF2PNG.py`](https://github.com/vintel38/RoofTop-Project/blob/master/doc/cvtTIF2PNG.py) disponible sur mon GitHub. Cependant, a priori, rien ne justifie la conversion `.tif` vers `.png`. De plus, il n'est mentionné nul part que Mask RCNN pourrait échouer pour des fichiers `.tif`. Aussi, pour résoudre tous ces problèmes et simplifier le procédé, les images satellites seront directement utilisées dans Mask RCNN au format `.tif`.  

- Le fait de créer des groupes augmentés d'images satellite en recadrant les fichiers pour qu'ils aient une taille de 1024x1024 px signifie également la perte de données géospatiales avec la bibliothèques généraliste OpenCV. De plus, ces recadrages sont réalisés avant la phase d'étiquettage ce qui signifie que l'étiquettage va être plus long et redondant sur des zones d'image qui se chevauchent entre les recadrages.  Cela mène à une augmentation du temps nécessaire pour la phase d'étiquettage déjà chronophage. Ce qui peut être fait pour résoudre ces deux problématiques est d'étiqueter directement les photos en taille 5000x5000 px pour éviter la redondance et d'appliquer le recadrage directement dans Mask RCNN au moment de l'entraînement ou de l'inférence avec des fonctions GDAL pour conserver le lien avec les données géospatiales ainsi qu'éviter la création de fichiers superflus. 

### Notes sur les problèmes pouvat être rencontrés :

Cependant, le fait d'étiqueter les images avant de les recadrer pose problème au moment du recadrage. En effet, les annotations sont décrites par des coordonnées locales sur les images en pixels. Il est donc (pour le moment) très complexe de recadrer les images après la phase d'annotation (qui sera fait dans les coordonnés globales utilisant la fonction `gdalwarp`) tout en conservant les annotations pertinentes en pixels. Dans un premier temps, la phase de recadrage restera avant la phase d'annotations car les images se chevauchant ne sont pas si nombreuses. 


## Phase 2 : Août 2021 - Implémentation avec GDAL
