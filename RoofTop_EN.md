# The RoofTop project : A satellite imagery and ML app fighting against climate change

In this article, I am going to present you a simple and thorough solution that I developed during my participation at Copernicus Masters 2021 organized by ESA. This hackathon promotes satellite imagery utilization by asking for concepts fighting against climate change. While using Mask-RCNN on imagery, the solution is able to locate buildings with dark roof which feed heat island phenomenon. 

<img src="https://github.com/vintel38/RoofTop/blob/master/doc/images/CM.png"  width="700" />

## The concept history : get lost on networks

Without the covid pandemic, I would never have participated in this competition. After I double-graduated in December 2020 in space propulsion, land my first job would not be easy. But mid-February 2021 while I was speaking with a friend and colleague of mine, we discussed a little bit about remote sensing and he mentioned the [Copernicus Masters](https://copernicus-masters.com/) competition. This hackathon immediately interested me as it was combining my interests in computer science, in Machine Learning I have been secretly learning for almost 2 years and my entrepreneurial aspirations. The covid19 pandemic also pushed me to question my professional plans. Indeed, I felt something has changed during this time. Suddenly, I was convinced that it was possible to help society without the need for more complex technology but by using differently the one which is already available to us and it is our duty, as engineers, to push to these changes. 

<img src="https://github.com/vintel38/RoofTop/blob/master/doc/images/Capture.png"  width="700" />

While scrolling on Facebook, only one video from [Loopsider](https://www.facebook.com/Loopsider/videos/500691364579511/) was enough for me to design the RoofTop pipeline. This short video presents a Breton start-up that started recently to paint roof with a reflective white painting, as Americans and Greeks have been doing for some time, to reflect solar rays. Through my two masters, my major stayed energetics but this solution was ridiculously confusing to me. If you don't want a premise to heat up, do not let heat enter it! I will explain in the next part the thermal principle. Because roofs are often flat and facing the sky, they can be observed by artificial satellites placed where comes the problem, ie solar rays. 

<img src="https://github.com/vintel38/RoofTop/blob/master/doc/images/chicago12_01.png"  width="700" />
Airborne very high resolution imagery of Chicago suburb from [Inria Aerial Image Labeling Dataset](https://project.inria.fr/aerialimagelabeling/)

Without thinking twice, I decided I will detect building dark roofs on very high-resolution imagery using Machine Learning algorithms. What is important to understand is the fact that this project does not limit itself to just a technical product but is a full package project. Clients and partners can be easily described and it gives key arguments to foster the process against the shadow competitor. Let me explain this in the next chapter.

## Physical principle and business opportunity

Painting roofs with a highly reflective painting implies a decrease in the building warm-up. Heat transfer inside buildings is divided into 3 sub-domains: conduction through solid materials, convection which comes with a fluid or gas macroscopic motion, and radiation which does not need matter to propagate. In the case of heat islands which are a direct consequence of global warming in urban areas, radiation is a major player that brings a surface solar power of roughly 1000 W/m^2 in broad daylight. This solar power will excite the molecules which are constituting the roof layer and will heat it. The painting solution is striking the roots of the problem. The highly reflective painting layer will mainly reflect the incident solar power. The solar radiation portion absorbed by the roof material diminishes so the temperature is less likely to increase inside the building. 

Roof with only isolation layers            |  Roof after isolation layer painting
:-------------------------:|:-------------------------:
<img src="https://github.com/vintel38/RoofTop/blob/master/doc/images/heat.png" width="300" /> | <img src="https://github.com/vintel38/RoofTop/blob/master/doc/images/cool.png"  width="300" />
Picture from the flyer of [CoolRoof](https://coolroof-france.com/wp-content/uploads/2021/05/plaquette_commerciale_en-1.pdf)  

White-painting roofs makes sense because of the smaller temperature increase in buildings. Consequently, you will use less air-conditioning which is not a famous device in terms of energy efficiency. This means tremendous electricity savings on the bill. It is therefore economically interesting to white-paint its roof and it also participates in fighting against global warming. Moreover, external roof layers will heat the surrounding atmosphere. In commercial areas, this process can explain a temperature increase up to several degrees. Because these commercial areas are often in the proximity of urban centers, they participate to the heat island phenomenon. Then, public institutions have a major interest in promoting the white-painting of commercial building roofs to improve the living conditions of their citizens.

<img src="https://github.com/vintel38/RoofTop/blob/master/doc/images/reserch_gate.png"  width="700" />
Urban heat island schematic representation (Source : Morris et Simmonds, 2000)

However, this process is still nascent in Europe but could save millions of tons of CO_2 emissions per year in electricity production. The goal of this project is to foster this process by producing a database of building coordinates that would benefit from roof white painting. This database can be exploited by public institutions or directly by the company exploiting the RoofTop pipeline to organize the white-painting process. Public institutions should subsidize it as it will improve their citizen's life conditions and decrease the electricity bill for all communities. 

## The pipeline 

The pipeline is eventually "pretty" simple with respect to the task it is solving. It is composed of 3 main parts using different technical bricks. You can see below a slide displaying the pipeline I presented during the Copernicus Masters competition.

<img src="https://github.com/vintel38/RoofTop/blob/master/doc/prez/3.jpg"  width="700" />

-Very high-resolution satellite imagery provided by space suppliers is ready-to-use, ie every atmospheric correction is already applied. During training, I am using [Inria Aerial Image Labeling Dataset](https://project.inria.fr/aerialimagelabeling/) to get a sufficient amount of VHR training set.
- Imagery is then pre-processed and formatted to be fed correctly to the Mask RCNN algorithm. More specifically, imagery resolution is decreased to 0.6 ground meter/pixel and is cut to only have square imagery of 1024 px side. Each step is performed using the GDAL library to preserve geospatial metadata embedded in the dataset *.tif* files.
- Preprocessed and split data into train/val/test subsets are then fed to the Mask RCNN Matterport algorithm. I chose it because it is one of the most efficient algorithm in the field of object detection/instance segmentation at the time being. This computer vision algorithm enables to detect (individually identify) specific elements of a picture and associates to these elements the pixels they belong to. Later, it will enable to estimate the roof surface to be white-painted. Using the image geospatial info, it is possible to compute the global GPS coordinates of each roof identified with its longitude and latitude.
- Each pair *(longitude, latitude)* is sent to the Google Maps API to be converted into addresses using reverse geocoding functionality. Information sent back by the API is then processed and stored in a PostGIS database to have a robust and flexible tool. Other API functionalities can be used to get more detailed information on the buildings like contact information.

Training is performed on a Google Colab session using a 12GB GPU even if I am considering buying my hardware to benefit from more powerful and more stable training processes. For confidentiality reasons, code associated with Google Colab and Google Maps API and their post-processing will not be disclosed nor the performance figures of the pipeline. Innovation contained in this project in the sense of the pipeline has been sent to INPI that is managing patent information in France. 

## Conclusion

At present, I have been able to test the pipeline on a few unseen images and the first results are quite encouraging. Even if some buildings are still not detected on 0.37 km^2 tiles, major building roofs are correctly identified by the AI and precisely located by the Google Maps API. Contrarily to the hackathon slide, TRL is currently more 6 than 5 because all pipeline is starting to perform well. For the data scientists among you, I am continuing to annotate tiles to boost pipeline performances to improve mAP and mAR of Mask RCNN.

The Copernicus Masters 2021 did not retain my project as the business model was not clear enough. On my side, I have already started to prospect potential incubators in the Aix-Marseille (FR) area which could host my project taking into account my professional constraints. My wish for the future of this project is to continue to add functionality and tools to fight against climate change from literature to end up with a complete SaaS solution to be proposed to development actors in a B2C or B2B business model. It would be a compromise between the know-how of a Kermap and the sense of innovation of a SpaceSense.

Keep in touch on my [LinkedIn](https://www.linkedin.com/in/vincent-arrigoni-8a80ba141/) and [Github](https://github.com/vintel38) for more deeptech projects. Find me on [UpWork](https://www.upwork.com/o/profiles/users/~01a4ac604eff45e6ae/) for your freelance work in computer vision. 